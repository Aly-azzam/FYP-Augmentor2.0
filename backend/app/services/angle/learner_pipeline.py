"""Learner angle extraction + DTW pipeline.

Orchestrates:
  1. Run YOLO + angle extraction on learner video (no annotated video written).
  2. Load pre-computed expert angles.json from disk.
  3. Run DTW and save alignment output.
  4. Save angles.json, dtw_alignment.json, learner_angle_comparison.json, run_summary.json.

Called by the app when a user uploads a video in Compare Studio.
"""

from __future__ import annotations

import json
import statistics
from pathlib import Path
from typing import Any

import cv2

from app.core.config import settings
from app.services.angle.detector import (
    ANGLE_JUMP_LIMIT_DEGREES,
    LOW_CONFIDENCE_ANGLE_THRESHOLD,
    angle_delta_degrees,
    angle_from_points,
    build_video_frames_payload,
    detect_scissors,
    estimate_blade_angle_from_crop,
    get_yolo_model,
    pick_best_scissors,
    smooth_angle,
)
from app.services.angle.dtw import (
    add_frame_indices_to_dtw_matches,
    build_dtw_alignment_payload,
    build_learner_comparison_payload,
    build_run_summary_payload,
    dtw_window_ratio,
    run_dtw,
)
from app.services.angle.preview import (
    LEARNER_COMPARISON_VIDEO_FILENAME,
    generate_learner_comparison_video,
)
from app.services.angle.schemas import (
    ANGLES_FILENAME,
    DTW_ALIGNMENT_FILENAME,
    LEARNER_COMPARISON_FILENAME,
    RUN_SUMMARY_FILENAME,
)


def run_learner_angle_pipeline(
    *,
    video_path: Path,
    run_id: str,
    expert_name: str,
    frame_stride: int | None = None,
) -> dict:
    """Run the full learner angle pipeline and return a summary dict."""
    stride = max(1, int(frame_stride or 1))
    angles_root = settings.ANGLES_OUTPUT_ROOT

    # ── Output directory ──────────────────────────────────────────────────────
    learner_out_dir = angles_root / "learner" / run_id
    learner_out_dir.mkdir(parents=True, exist_ok=True)

    # ── Step 1: YOLO + angle extraction on learner video ──────────────────────
    print(f"Processing learner video: {video_path.name}")
    learner_frames_payload = _extract_angles_from_video(
        video_path=video_path,
        video_type="learner",
        frame_stride=stride,
    )
    angles_path = learner_out_dir / ANGLES_FILENAME
    _write_json(angles_path, learner_frames_payload)
    print(f"Saved {ANGLES_FILENAME}")

    # ── Step 2: Load expert angles ────────────────────────────────────────────
    expert_angles_path = angles_root / "expert" / expert_name / ANGLES_FILENAME
    if not expert_angles_path.is_file():
        raise FileNotFoundError(
            f"Expert angles not found. Run offline script first. "
            f"Expected: {expert_angles_path}"
        )
    expert_frames_payload = json.loads(expert_angles_path.read_text(encoding="utf-8"))

    # ── Step 3: DTW ───────────────────────────────────────────────────────────
    expert_valid_frames = extract_valid_line_frames(expert_frames_payload)
    learner_valid_frames = extract_valid_line_frames(learner_frames_payload)

    expert_angles = [frame["line_angle"] for frame in expert_valid_frames]
    learner_angles = [frame["line_angle"] for frame in learner_valid_frames]

    print("Running DTW...")
    dtw_result = run_dtw(
        expert_angles,
        learner_angles,
        window_ratio=dtw_window_ratio(expert_angles, learner_angles),
    )
    dtw_result = add_frame_indices_to_dtw_matches(
        dtw_result,
        expert_frames=expert_valid_frames,
        learner_frames=learner_valid_frames,
    )
    dtw_alignment_payload = build_dtw_alignment_payload(
        dtw_result,
        expert_valid_frame_count=len(expert_valid_frames),
        learner_valid_frame_count=len(learner_valid_frames),
    )

    dtw_path = learner_out_dir / DTW_ALIGNMENT_FILENAME
    _write_json(dtw_path, dtw_alignment_payload)
    print(f"Saved {DTW_ALIGNMENT_FILENAME}")

    # ── Step 4: Learner comparison + run summary ──────────────────────────────
    learner_comparison_payload = build_learner_comparison_payload(
        learner_frames_payload=learner_frames_payload,
        dtw_alignment_payload=dtw_alignment_payload,
    )
    comparison_path = learner_out_dir / LEARNER_COMPARISON_FILENAME
    _write_json(comparison_path, learner_comparison_payload)
    print(f"Saved {LEARNER_COMPARISON_FILENAME}")

    run_summary = build_run_summary_payload(
        clip_id=run_id,
        expert_name=expert_name,
        expert_frames_payload=expert_frames_payload,
        learner_frames_payload=learner_frames_payload,
        dtw_alignment_payload=dtw_alignment_payload,
        learner_comparison_payload=learner_comparison_payload,
    )
    summary_path = learner_out_dir / RUN_SUMMARY_FILENAME
    _write_json(summary_path, run_summary)
    print(f"Saved {RUN_SUMMARY_FILENAME}")

    # ── Step 5: Generate learner_comparison_output.mp4 ────────────────────────
    comparison_video_path = learner_out_dir / LEARNER_COMPARISON_VIDEO_FILENAME
    print("Generating learner_comparison_output.mp4...")
    try:
        generate_learner_comparison_video(
            learner_video_path=video_path,
            learner_comparison_payload=learner_comparison_payload,
            output_path=comparison_video_path,
        )
        print(f"Saved {LEARNER_COMPARISON_VIDEO_FILENAME}")
    except Exception as exc:
        print(f"WARNING: Could not generate comparison video: {exc}")
        comparison_video_path = None

    return {
        "run_id": run_id,
        "expert_name": expert_name,
        "angles_path": str(angles_path),
        "dtw_alignment_path": str(dtw_path),
        "learner_comparison_path": str(comparison_path),
        "run_summary_path": str(summary_path),
        "learner_comparison_video_path": str(comparison_video_path) if comparison_video_path else None,
        "normalized_dtw_distance": run_summary["normalized_dtw_distance"],
        "mean_angle_difference": run_summary["mean_angle_difference"],
        "median_angle_difference": run_summary["median_angle_difference"],
        "max_angle_difference": run_summary["max_angle_difference"],
    }


# ── Core angle extraction (no video writing) ──────────────────────────────────

def _extract_angles_from_video(
    *,
    video_path: Path,
    video_type: str,
    frame_stride: int,
) -> dict:
    """Run YOLO + angle extraction on a video and return the frames payload dict.

    Does NOT write an annotated video — only extracts angle data.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if width <= 0 or height <= 0:
        cap.release()
        raise RuntimeError(f"Invalid video dimensions for {video_path}")

    model = get_yolo_model()
    frames: list[dict] = []
    frame_index = 0
    last_bbox: list[float] | None = None
    last_confidence = 0.0
    previous_angle: float | None = None

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            status = "skipped"
            bbox = None
            confidence = 0.0
            line_center = None
            line_angle = None
            line_start = None
            line_end = None
            line_source = "none"
            fallback_used = False
            valid_line = False

            if frame_index % frame_stride == 0:
                predictions = detect_scissors(frame, model)
                picked, _ = pick_best_scissors(predictions, width, height)

                if picked is not None:
                    bbox = [float(v) for v in picked["bbox"]]
                    confidence = float(picked.get("confidence", 0.0))
                    last_bbox = bbox
                    last_confidence = confidence
                    status = "detected"
                else:
                    status = "no_detection"
            else:
                if last_bbox is not None:
                    bbox = last_bbox
                    confidence = last_confidence
                    status = "reused"

            if bbox is not None:
                x1, y1, x2, y2 = map(int, bbox)
                center_x = (x1 + x2) / 2.0
                center_y = (y1 + y2) / 2.0
                line_center = [round(center_x, 3), round(center_y, 3)]

                bbox_angle = angle_from_points((x1, y1), (x2, y2))
                blade_angle = estimate_blade_angle_from_crop(frame, (x1, y1, x2, y2))

                if previous_angle is None:
                    if blade_angle is not None:
                        angle_to_draw = blade_angle
                        line_source = "fitline"
                    else:
                        angle_to_draw = bbox_angle
                        line_source = "bbox_diagonal"
                        fallback_used = True
                elif blade_angle is None or confidence < LOW_CONFIDENCE_ANGLE_THRESHOLD:
                    angle_to_draw = previous_angle
                    line_source = "previous_frame"
                    fallback_used = True
                else:
                    jump = abs(angle_delta_degrees(previous_angle, blade_angle))
                    if jump > ANGLE_JUMP_LIMIT_DEGREES:
                        angle_to_draw = previous_angle
                        line_source = "previous_frame"
                        fallback_used = True
                    else:
                        angle_to_draw = smooth_angle(previous_angle, blade_angle)
                        line_source = "fitline"

                from app.services.angle.detector import extend_angle_line

                start, end = extend_angle_line(angle_to_draw, center_x, center_y, width, height)
                line_start = [int(start[0]), int(start[1])]
                line_end = [int(end[0]), int(end[1])]
                previous_angle = angle_to_draw
                line_angle = round(float(angle_to_draw), 6)
                valid_line = True

            frames.append(
                {
                    "frame_index": frame_index,
                    "bbox": bbox,
                    "confidence": round(confidence, 6),
                    "line_center": line_center,
                    "line_angle": line_angle,
                    "line_start": line_start,
                    "line_end": line_end,
                    "line_source": line_source,
                    "fallback_used": fallback_used,
                    "valid_line": valid_line,
                    "status": status,
                }
            )

            frame_index += 1

    finally:
        cap.release()

    if frame_index == 0:
        raise RuntimeError(f"No frames were processed from {video_path}")

    return build_video_frames_payload(
        video_type=video_type,
        video_path=str(video_path),
        total_frames=frame_index,
        fps=float(fps),
        frames=frames,
    )


# ── Helpers ────────────────────────────────────────────────────────────────────

def extract_valid_line_frames(payload: dict) -> list[dict]:
    """Return only frames with a valid detected angle, as lightweight dicts."""
    frames = [
        {
            "frame_index": int(frame["frame_index"]),
            "timestamp_sec": float(frame.get("timestamp_sec", 0.0)),
            "line_angle": float(frame["line_angle"]),
        }
        for frame in payload.get("frames", [])
        if frame.get("valid_line") and frame.get("line_angle") is not None
    ]
    if not frames:
        raise ValueError("No valid line angles found in the provided payload")
    return frames


def build_learner_angles_payload(
    *,
    video_path: str,
    total_frames: int,
    fps: float,
    frames: list[dict],
) -> dict:
    """Alias kept for external callers that import this name explicitly."""
    return build_video_frames_payload(
        video_type="learner",
        video_path=video_path,
        total_frames=total_frames,
        fps=fps,
        frames=frames,
    )


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
