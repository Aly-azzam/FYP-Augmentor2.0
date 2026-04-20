"""MediaPipe visualization service.

Consumes ``detections.json`` and ``features.json`` from a run folder and
renders an annotated ``.mp4`` to make the hand tracking visible. The
video is written back into the same run folder:

    backend/storage/mediapipe/runs/<run_id>/annotated.mp4

Overlays drawn per frame:
    * full 21 hand landmarks + MediaPipe Hands skeleton
    * strong wrist marker
    * index / thumb / middle tip markers (distinct colors)
    * hand center marker
    * hand bounding box rectangle
    * wrist trajectory history trail (polyline)
    * handedness + confidence text (optional)

All coordinates in the JSON documents are stored in MediaPipe's normalized
[0, 1] image space and must be multiplied by the frame width/height here
before drawing.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

from app.schemas.mediapipe.mediapipe_schema import (
    MediaPipeDetectionsDocument,
    MediaPipeFeaturesDocument,
    MediaPipeFrameFeatures,
    MediaPipeFrameRaw,
)


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Drawing constants (BGR because we render on OpenCV frames).
# ---------------------------------------------------------------------------

HAND_SKELETON_CONNECTIONS: List[Tuple[int, int]] = [
    # thumb
    (0, 1), (1, 2), (2, 3), (3, 4),
    # index
    (0, 5), (5, 6), (6, 7), (7, 8),
    # middle
    (5, 9), (9, 10), (10, 11), (11, 12),
    # ring
    (9, 13), (13, 14), (14, 15), (15, 16),
    # pinky
    (13, 17), (17, 18), (18, 19), (19, 20),
    # palm base
    (0, 17),
]

# Colors in BGR.
COLOR_SKELETON = (200, 200, 200)        # light gray
COLOR_LANDMARK = (0, 255, 255)          # yellow
COLOR_WRIST = (0, 0, 255)               # red
COLOR_WRIST_RING = (0, 0, 255)
COLOR_INDEX_TIP = (255, 255, 0)         # cyan
COLOR_THUMB_TIP = (0, 255, 0)           # green
COLOR_MIDDLE_TIP = (255, 0, 255)        # magenta
COLOR_HAND_CENTER = (0, 180, 255)       # orange
COLOR_BBOX = (0, 255, 0)                # green
COLOR_TRAIL = (255, 160, 0)             # blue-orange trail
COLOR_TEXT = (255, 255, 255)            # white
COLOR_TEXT_BG = (0, 0, 0)               # black

_DEFAULT_FPS = 30.0
_ANNOTATED_FILENAME = "annotated.mp4"
_VIDEO_CODEC_CANDIDATES: List[str] = ["avc1", "H264", "X264", "mp4v", "FMP4"]


class MediaPipeVisualizationError(RuntimeError):
    """Raised when the annotated video cannot be produced."""


def _open_video_writer(
    output_path: Path,
    *,
    fps: float,
    width: int,
    height: int,
) -> cv2.VideoWriter:
    """Open a VideoWriter using the first available codec.

    We prefer H.264-compatible tags first because browser playback is more
    reliable for those streams. If unavailable on a host, we gracefully
    fall back to MPEG-4 tags.
    """
    for codec in _VIDEO_CODEC_CANDIDATES:
        fourcc = cv2.VideoWriter_fourcc(*codec)
        writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
        if writer.isOpened():
            logger.info("MediaPipe annotation writer codec selected: %s", codec)
            return writer
        writer.release()

    raise MediaPipeVisualizationError(
        f"OpenCV VideoWriter could not be opened for: {output_path}"
    )


# ---------------------------------------------------------------------------
# Drawing primitives
# ---------------------------------------------------------------------------

def _denormalize_point(
    normalized: Optional[List[float]],
    width: int,
    height: int,
) -> Optional[Tuple[int, int]]:
    if normalized is None or len(normalized) < 2:
        return None
    x = int(round(float(normalized[0]) * width))
    y = int(round(float(normalized[1]) * height))
    return (x, y)


def _draw_skeleton(
    frame: np.ndarray,
    landmarks_xy: List[Tuple[int, int]],
) -> None:
    for start_index, end_index in HAND_SKELETON_CONNECTIONS:
        if start_index >= len(landmarks_xy) or end_index >= len(landmarks_xy):
            continue
        start_point = landmarks_xy[start_index]
        end_point = landmarks_xy[end_index]
        cv2.line(frame, start_point, end_point, COLOR_SKELETON, thickness=2, lineType=cv2.LINE_AA)

    for point in landmarks_xy:
        cv2.circle(frame, point, radius=3, color=COLOR_LANDMARK, thickness=-1, lineType=cv2.LINE_AA)


def _draw_strong_marker(
    frame: np.ndarray,
    center: Optional[Tuple[int, int]],
    *,
    color: Tuple[int, int, int],
    inner_radius: int = 6,
    outer_radius: int = 12,
) -> None:
    if center is None:
        return
    cv2.circle(frame, center, outer_radius, color, thickness=2, lineType=cv2.LINE_AA)
    cv2.circle(frame, center, inner_radius, color, thickness=-1, lineType=cv2.LINE_AA)


def _draw_bbox(
    frame: np.ndarray,
    bbox,
    width: int,
    height: int,
) -> None:
    if bbox is None:
        return
    x_min = int(round(bbox.x_min * width))
    y_min = int(round(bbox.y_min * height))
    x_max = int(round(bbox.x_max * width))
    y_max = int(round(bbox.y_max * height))
    cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), COLOR_BBOX, thickness=2, lineType=cv2.LINE_AA)


def _draw_trajectory(
    frame: np.ndarray,
    trajectory: List[List[float]],
    width: int,
    height: int,
) -> None:
    if not trajectory or len(trajectory) < 2:
        return
    points = np.array(
        [[int(round(x * width)), int(round(y * height))] for x, y in trajectory],
        dtype=np.int32,
    )
    cv2.polylines(frame, [points], isClosed=False, color=COLOR_TRAIL, thickness=2, lineType=cv2.LINE_AA)
    # Emphasize the most recent trail point.
    cv2.circle(frame, tuple(points[-1]), radius=4, color=COLOR_TRAIL, thickness=-1, lineType=cv2.LINE_AA)


def _draw_text_block(
    frame: np.ndarray,
    lines: List[str],
    *,
    origin: Tuple[int, int] = (12, 24),
    font_scale: float = 0.6,
    thickness: int = 1,
) -> None:
    if not lines:
        return
    font = cv2.FONT_HERSHEY_SIMPLEX
    line_gap = int(28 * font_scale / 0.6)
    padding = 6
    sizes = [cv2.getTextSize(line, font, font_scale, thickness)[0] for line in lines]
    max_width = max(width for width, _ in sizes)
    total_height = line_gap * len(lines)
    x0, y0 = origin
    cv2.rectangle(
        frame,
        (x0 - padding, y0 - line_gap + 4),
        (x0 + max_width + padding, y0 - line_gap + 4 + total_height + padding),
        COLOR_TEXT_BG,
        thickness=-1,
    )
    for index, line in enumerate(lines):
        y = y0 + index * line_gap
        cv2.putText(frame, line, (x0, y), font, font_scale, COLOR_TEXT, thickness, cv2.LINE_AA)


# ---------------------------------------------------------------------------
# Per-frame overlay
# ---------------------------------------------------------------------------

def _draw_overlay(
    frame: np.ndarray,
    raw_frame: MediaPipeFrameRaw,
    feature_frame: Optional[MediaPipeFrameFeatures],
    *,
    width: int,
    height: int,
    show_text: bool,
) -> None:
    # Trajectory is cheap to draw even when the current frame has no
    # detection; it reflects the most recent wrist positions.
    if feature_frame is not None:
        _draw_trajectory(frame, feature_frame.trajectory_history, width, height)

    if not raw_frame.has_detection or raw_frame.selected_hand is None:
        if show_text:
            _draw_text_block(
                frame,
                [f"frame {raw_frame.frame_index}", "no hand detected"],
            )
        return

    selected = raw_frame.selected_hand

    landmarks_xy: List[Tuple[int, int]] = []
    for landmark in selected.landmarks:
        landmarks_xy.append(
            (
                int(round(landmark.x * width)),
                int(round(landmark.y * height)),
            )
        )

    if feature_frame is not None and feature_frame.hand_bbox is not None:
        _draw_bbox(frame, feature_frame.hand_bbox, width, height)

    _draw_skeleton(frame, landmarks_xy)

    wrist_xy = _denormalize_point(selected.wrist, width, height)
    index_xy = _denormalize_point(selected.index_tip, width, height)
    thumb_xy = _denormalize_point(selected.thumb_tip, width, height)
    middle_xy = _denormalize_point(selected.middle_tip, width, height)
    center_xy = _denormalize_point(selected.hand_center, width, height)

    _draw_strong_marker(frame, wrist_xy, color=COLOR_WRIST, inner_radius=7, outer_radius=14)
    _draw_strong_marker(frame, index_xy, color=COLOR_INDEX_TIP, inner_radius=5, outer_radius=10)
    _draw_strong_marker(frame, thumb_xy, color=COLOR_THUMB_TIP, inner_radius=5, outer_radius=10)
    _draw_strong_marker(frame, middle_xy, color=COLOR_MIDDLE_TIP, inner_radius=5, outer_radius=10)
    _draw_strong_marker(frame, center_xy, color=COLOR_HAND_CENTER, inner_radius=4, outer_radius=8)

    if show_text:
        lines = [
            f"frame {raw_frame.frame_index}",
            f"hand: {selected.handedness}  conf: {selected.detection_confidence:.2f}",
        ]
        if feature_frame is not None and feature_frame.hand_orientation_deg is not None:
            lines.append(f"orientation: {feature_frame.hand_orientation_deg:.1f} deg")
        _draw_text_block(frame, lines)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def render_annotated_video(
    run_dir: str | Path,
    *,
    source_video_path: Optional[str | Path] = None,
    output_filename: str = _ANNOTATED_FILENAME,
    show_text: bool = True,
) -> Path:
    """Render ``annotated.mp4`` for a MediaPipe run.

    ``run_dir`` must already contain ``detections.json`` and
    ``features.json``. If ``source_video_path`` is not supplied, it is read
    from ``detections.json``.
    """
    run_path = Path(run_dir).resolve()
    detections_path = run_path / "detections.json"
    features_path = run_path / "features.json"

    if not detections_path.is_file():
        raise MediaPipeVisualizationError(
            f"detections.json missing from run folder: {detections_path}"
        )
    if not features_path.is_file():
        raise MediaPipeVisualizationError(
            f"features.json missing from run folder: {features_path}"
        )

    with detections_path.open("r", encoding="utf-8") as handle:
        detections_document = MediaPipeDetectionsDocument.model_validate(json.load(handle))
    with features_path.open("r", encoding="utf-8") as handle:
        features_document = MediaPipeFeaturesDocument.model_validate(json.load(handle))

    video_path = Path(source_video_path or detections_document.source_video_path).resolve()
    if not video_path.is_file():
        raise MediaPipeVisualizationError(
            f"Source video not found for annotation: {video_path}"
        )

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise MediaPipeVisualizationError(f"OpenCV could not open video: {video_path}")

    fps = float(detections_document.fps) if detections_document.fps > 0 else _DEFAULT_FPS
    width = int(detections_document.width)
    height = int(detections_document.height)

    # Index feature frames by frame_index for O(1) lookup even when the
    # feature document is produced out of order (it isn't, but this keeps
    # the code robust to that possibility).
    feature_by_index = {feat.frame_index: feat for feat in features_document.frames}
    raw_by_index = {raw.frame_index: raw for raw in detections_document.frames}

    output_path = run_path / output_filename
    try:
        writer = _open_video_writer(output_path, fps=fps, width=width, height=height)
    except Exception:
        capture.release()
        raise

    frames_rendered = 0
    try:
        frame_index = 0
        while True:
            success, bgr_frame = capture.read()
            if not success or bgr_frame is None:
                break

            if bgr_frame.shape[1] != width or bgr_frame.shape[0] != height:
                bgr_frame = cv2.resize(bgr_frame, (width, height), interpolation=cv2.INTER_AREA)

            raw_frame = raw_by_index.get(frame_index)
            if raw_frame is not None:
                feature_frame = feature_by_index.get(frame_index)
                _draw_overlay(
                    bgr_frame,
                    raw_frame,
                    feature_frame,
                    width=width,
                    height=height,
                    show_text=show_text,
                )
            elif show_text:
                _draw_text_block(bgr_frame, [f"frame {frame_index}", "no detection record"])

            writer.write(bgr_frame)
            frames_rendered += 1
            frame_index += 1
    finally:
        capture.release()
        writer.release()

    logger.info(
        "MediaPipe annotation complete: run_id=%s frames=%s output=%s",
        detections_document.run_id,
        frames_rendered,
        output_path,
    )
    return output_path
