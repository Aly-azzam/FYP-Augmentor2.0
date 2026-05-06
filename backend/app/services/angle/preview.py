"""Video preview generators for the angle pipeline.

Two functions live here:

1. generate_learner_comparison_video()
   Renders learner_comparison_output.mp4 — the learner video annotated with:
     - Red YOLO bounding box
     - Green learner angle line
     - Blue expert reference angle line
     - Cyan arc showing the angular difference
     - Top-left text: "Learner angle / Expert ref angle / Difference"
   Called by learner_pipeline.py as part of the normal pipeline run.

2. create_dtw_aligned_preview()
   Renders dtw_aligned_preview.mp4 — side-by-side expert + learner frames
   synced by DTW alignment.
   Called ONLY on-demand when the user clicks the SYNC button.

Drawing logic copied from yolo_scissors_test/app/main.py.
"""

from __future__ import annotations

import math
import subprocess
from pathlib import Path
from typing import Any

import cv2
import numpy as np

LEARNER_COMPARISON_VIDEO_FILENAME = "learner_comparison_output.mp4"


# ══════════════════════════════════════════════════════════════════════════════
# Learner comparison video
# ══════════════════════════════════════════════════════════════════════════════

def generate_learner_comparison_video(
    *,
    learner_video_path: Path,
    learner_comparison_payload: dict,
    output_path: Path,
) -> None:
    """Render the annotated learner comparison video.

    For every frame in the learner video, draw:
    - Red YOLO bounding box + confidence label
    - Green line for the learner's detected angle
    - Blue line for the matched expert reference angle
    - Cyan arc between the two lines
    - Top-left text overlay:
        "Learner angle: X deg"
        "Expert ref angle: X deg"
        "Difference: X deg"

    Args:
        learner_video_path: Path to the raw learner input video.
        learner_comparison_payload: The dict saved as learner_angle_comparison.json,
            containing a "frames" list keyed by frame_index.
        output_path: Where to write the final H.264 MP4.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_avi = output_path.with_suffix(".tmp.avi")
    tmp_mp4 = output_path.with_suffix(".tmp.mp4")

    # Build a fast lookup: frame_index → frame record
    frame_map: dict[int, dict] = {
        int(f["frame_index"]): f
        for f in learner_comparison_payload.get("frames", [])
    }

    cap = cv2.VideoCapture(str(learner_video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open learner video: {learner_video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if width <= 0 or height <= 0:
        cap.release()
        raise RuntimeError(f"Invalid video dimensions for {learner_video_path}")

    writer = cv2.VideoWriter(
        str(tmp_avi),
        cv2.VideoWriter_fourcc(*"MJPG"),
        float(fps),
        (width, height),
    )
    if not writer.isOpened():
        cap.release()
        raise RuntimeError("Could not create learner comparison video writer")

    frame_index = 0
    written = 0

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            rec = frame_map.get(frame_index)
            if rec is not None:
                _draw_learner_comparison_frame(
                    frame,
                    rec=rec,
                    width=width,
                    height=height,
                )

            writer.write(frame)
            written += 1
            frame_index += 1
    finally:
        cap.release()
        writer.release()

    if written == 0:
        tmp_avi.unlink(missing_ok=True)
        raise RuntimeError("No frames written to learner comparison video")

    try:
        transcode_avi_mjpeg_to_h264_mp4(input_avi=tmp_avi, output_mp4=tmp_mp4, fps=float(fps))
    except RuntimeError:
        tmp_avi.unlink(missing_ok=True)
        tmp_mp4.unlink(missing_ok=True)
        raise

    tmp_avi.unlink(missing_ok=True)
    if output_path.exists():
        output_path.unlink()
    tmp_mp4.replace(output_path)


def _draw_learner_comparison_frame(
    frame: Any,
    *,
    rec: dict,
    width: int,
    height: int,
) -> None:
    """Draw all annotations onto a single learner frame in-place."""
    from app.services.angle.detector import extend_angle_line
    from app.services.angle.dtw import angle_difference_degrees

    learner_angle: float | None = rec.get("learner_angle")
    expert_ref_angle: float | None = rec.get("matched_expert_angle")
    learner_center = rec.get("learner_line_center")  # {"x": ..., "y": ...}

    # Determine center from line_center or bbox fallback
    if learner_center and isinstance(learner_center, dict):
        center_x = float(learner_center.get("x", width / 2))
        center_y = float(learner_center.get("y", height / 2))
    else:
        center_x = width / 2.0
        center_y = height / 2.0

    # ── Bounding box (red) ────────────────────────────────────────────────────
    # learner_line_center is the scissors center; we don't store bbox in
    # comparison payload, so only draw center dot if no bbox info.
    # (The bbox was not persisted in comparison payload — skip rectangle.)

    # ── Learner angle line (green) ─────────────────────────────────────────
    if rec.get("learner_valid_line") and learner_angle is not None:
        start, end = extend_angle_line(learner_angle, center_x, center_y, width, height)
        cv2.line(frame, start, end, (0, 255, 0), 4)

    # ── Expert reference line (blue) ──────────────────────────────────────
    if expert_ref_angle is not None:
        ref_start, ref_end = extend_angle_line(
            expert_ref_angle, center_x, center_y, width, height
        )
        cv2.line(frame, ref_start, ref_end, (255, 0, 0), 3)

    # ── Arc + difference text + top-left stats ────────────────────────────
    if (
        rec.get("learner_valid_line")
        and learner_angle is not None
        and expert_ref_angle is not None
    ):
        diff = angle_difference_degrees(learner_angle, expert_ref_angle)

        draw_angle_difference_arc(
            frame,
            center=(center_x, center_y),
            learner_angle=learner_angle,
            expert_reference_angle=expert_ref_angle,
            angle_difference=diff,
        )

        draw_learner_compare_text(
            frame,
            learner_angle=learner_angle,
            expert_reference_angle=expert_ref_angle,
            angle_difference=diff,
        )


# ── Drawing helpers (copied exactly from yolo_scissors_test/app/main.py) ─────

def draw_learner_compare_text(
    frame: Any,
    *,
    learner_angle: float,
    expert_reference_angle: float,
    angle_difference: float,
) -> None:
    # ── Stats (blue) ─────────────────────────────────────────────────────────
    stats = [
        f"Learner angle: {learner_angle:.1f} deg",
        f"Expert ref angle: {expert_reference_angle:.1f} deg",
        f"Difference: {angle_difference:.1f} deg",
    ]
    y = 28
    for line in stats:
        cv2.putText(
            frame, line, (18, y),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2, cv2.LINE_AA,
        )
        y += 28

    # ── Legend — each label drawn in the color of its line ───────────────────
    # small gap before legend
    y += 8
    legend = [
        ("Expert line  : blue",   (255,   0,   0)),   # BGR blue  = expert ref line
        ("Learner line : green",  (  0, 255,   0)),   # BGR green = learner angle line
    ]
    for label, color in legend:
        cv2.putText(
            frame, label, (18, y),
            cv2.FONT_HERSHEY_SIMPLEX, 0.62, color, 2, cv2.LINE_AA,
        )
        y += 26


def draw_angle_difference_arc(
    frame: Any,
    *,
    center: tuple[float, float],
    learner_angle: float,
    expert_reference_angle: float,
    angle_difference: float,
) -> None:
    center_point = (int(round(center[0])), int(round(center[1])))
    text_position = (center_point[0] + 70, max(24, center_point[1] - 10))

    if angle_difference > 3:
        visual_diff = _draw_small_angle_arc(
            frame,
            center=center_point,
            expert_angle=expert_reference_angle,
            learner_angle=learner_angle,
        )
        cv2.putText(
            frame,
            f"Angle diff: {visual_diff:.1f} deg",
            text_position,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )
    else:
        cv2.putText(
            frame,
            f"Aligned: {angle_difference:.1f} deg",
            text_position,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )


def _draw_small_angle_arc(
    frame: Any,
    center: tuple[int, int],
    expert_angle: float,
    learner_angle: float,
    radius: int = 55,
    steps: int = 25,
) -> float:
    a1 = expert_angle % 180
    a2 = learner_angle % 180

    diff = ((a2 - a1 + 90) % 180) - 90
    start = a1
    points = _make_arc_points(center, start, diff, radius=radius, steps=steps)

    avg_y = sum(p[1] for p in points) / len(points)
    if avg_y > center[1]:
        points = _make_arc_points(center, start + 180, diff, radius=radius, steps=steps)

    for p1, p2 in zip(points[:-1], points[1:]):
        cv2.line(frame, p1, p2, (0, 255, 255), 4)

    cv2.line(frame, center, points[0], (0, 255, 255), 2)
    cv2.line(frame, center, points[-1], (0, 255, 255), 2)

    return abs(diff)


def _make_arc_points(
    center: tuple[int, int],
    start_angle: float,
    diff: float,
    radius: int = 60,
    steps: int = 30,
) -> list[tuple[int, int]]:
    points = []
    for k in range(steps + 1):
        t = k / steps
        angle_radians = math.radians(start_angle + diff * t)
        x = int(center[0] + radius * math.cos(angle_radians))
        y = int(center[1] + radius * math.sin(angle_radians))
        points.append((x, y))
    return points


# ══════════════════════════════════════════════════════════════════════════════
# DTW aligned preview (SYNC button only)
# ══════════════════════════════════════════════════════════════════════════════

def create_dtw_aligned_preview(
    *,
    expert_video_path: Path,
    learner_video_path: Path,
    dtw_matches: list[dict],
    output_path: Path,
    fps: float = 10.0,
) -> None:
    if not dtw_matches:
        raise RuntimeError("DTW returned no matches for preview")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_avi_path = output_path.with_suffix(".tmp.avi")
    tmp_mp4_path = output_path.with_suffix(".tmp.mp4")

    expert_cap = cv2.VideoCapture(str(expert_video_path))
    learner_cap = cv2.VideoCapture(str(learner_video_path))
    if not expert_cap.isOpened() or not learner_cap.isOpened():
        expert_cap.release()
        learner_cap.release()
        raise RuntimeError("Could not open annotated videos for DTW preview")

    writer = None
    written_frames = 0
    try:
        for step_index, match in enumerate(dtw_matches):
            expert_frame = read_video_frame(expert_cap, int(match["expert_frame_index"]))
            learner_frame = read_video_frame(learner_cap, int(match["learner_frame_index"]))
            if expert_frame is None or learner_frame is None:
                continue

            combined = make_dtw_preview_frame(
                expert_frame=expert_frame,
                learner_frame=learner_frame,
                match=match,
                step_index=step_index,
            )
            if writer is None:
                height, width = combined.shape[:2]
                writer = cv2.VideoWriter(
                    str(tmp_avi_path),
                    cv2.VideoWriter_fourcc(*"MJPG"),
                    fps,
                    (width, height),
                )
                if not writer.isOpened():
                    raise RuntimeError("Could not create DTW preview writer")

            writer.write(combined)
            written_frames += 1
    finally:
        expert_cap.release()
        learner_cap.release()
        if writer is not None:
            writer.release()

    if written_frames == 0:
        if tmp_avi_path.exists():
            tmp_avi_path.unlink()
        raise RuntimeError("No DTW preview frames were written")

    try:
        transcode_avi_mjpeg_to_h264_mp4(input_avi=tmp_avi_path, output_mp4=tmp_mp4_path, fps=fps)
    except RuntimeError:
        if tmp_avi_path.exists():
            tmp_avi_path.unlink()
        if tmp_mp4_path.exists():
            tmp_mp4_path.unlink()
        raise

    if tmp_avi_path.exists():
        tmp_avi_path.unlink()
    if output_path.exists():
        output_path.unlink()
    tmp_mp4_path.replace(output_path)


def read_video_frame(cap: cv2.VideoCapture, frame_index: int) -> Any | None:
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ok, frame = cap.read()
    if not ok:
        return None
    return frame


def make_dtw_preview_frame(
    *,
    expert_frame: Any,
    learner_frame: Any,
    match: dict,
    step_index: int,
) -> Any:
    target_height = min(expert_frame.shape[0], learner_frame.shape[0])
    expert_frame = _resize_to_height(expert_frame, target_height)
    learner_frame = _resize_to_height(learner_frame, target_height)
    combined = np.hstack([expert_frame, learner_frame])

    overlay_lines = [
        f"DTW step: {step_index}",
        f"Expert frame: {match['expert_frame_index']}",
        f"Learner frame: {match['learner_frame_index']}",
        f"Expert angle: {float(match['expert_angle']):.1f} deg",
        f"Learner angle: {float(match['learner_angle']):.1f} deg",
        f"Difference: {float(match['angle_difference']):.1f} deg",
    ]
    y = 30
    for line in overlay_lines:
        cv2.putText(
            combined, line, (18, y),
            cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 3, cv2.LINE_AA,
        )
        cv2.putText(
            combined, line, (18, y),
            cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 1, cv2.LINE_AA,
        )
        y += 30

    split_x = expert_frame.shape[1]
    cv2.line(combined, (split_x, 0), (split_x, combined.shape[0]), (255, 255, 255), 2)
    cv2.putText(
        combined, "Expert", (18, combined.shape[0] - 20),
        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2,
    )
    cv2.putText(
        combined, "Learner", (split_x + 18, combined.shape[0] - 20),
        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2,
    )
    return _ensure_even_frame_size(combined)


# ── Shared utilities ──────────────────────────────────────────────────────────

def transcode_avi_mjpeg_to_h264_mp4(*, input_avi: Path, output_mp4: Path, fps: float) -> None:
    import imageio_ffmpeg

    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    if output_mp4.exists():
        output_mp4.unlink()

    cmd = [
        ffmpeg, "-y", "-hide_banner", "-loglevel", "error",
        "-i", str(input_avi),
        "-an", "-c:v", "libx264", "-pix_fmt", "yuv420p",
        "-preset", "veryfast", "-crf", "23",
        "-movflags", "+faststart",
        "-r", str(fps),
        str(output_mp4),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"FFmpeg failed (exit {result.returncode}): {result.stderr[:500]}"
        )


def _resize_to_height(frame: Any, target_height: int) -> Any:
    height, width = frame.shape[:2]
    if height == target_height:
        return frame
    scale = target_height / height
    target_width = max(1, int(round(width * scale)))
    return cv2.resize(frame, (target_width, target_height), interpolation=cv2.INTER_AREA)


def _ensure_even_frame_size(frame: Any) -> Any:
    height, width = frame.shape[:2]
    even_height = height - (height % 2)
    even_width = width - (width % 2)
    if even_height == height and even_width == width:
        return frame
    return frame[:even_height, :even_width]
