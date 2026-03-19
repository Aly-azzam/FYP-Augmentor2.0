"""Perception Engine: video preprocessing.

This module is responsible for preparing video data for later stages
(e.g., MediaPipe landmark extraction). It provides:
- video loading
- video metadata reading
- frame extraction (with optional FPS downsampling)
- optional frame resizing
- OpenCV BGR -> RGB conversion
"""

from __future__ import annotations

import os
import math
from typing import Any, Dict, List

import cv2


def get_video_metadata(video_path: str) -> dict:
    """Return basic metadata for a video file.

    Args:
        video_path: Path to the input video file.

    Returns:
        Dict with `video_path`, `fps`, `total_frames`, `width`, `height`, and
        `duration_seconds`.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If OpenCV cannot open the file.
    """
    if not os.path.isfile(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")

    cap = cv2.VideoCapture(video_path)
    try:
        if not cap.isOpened():
            raise ValueError(f"Could not open video for reading: {video_path}")

        fps_raw = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        fps = fps_raw if (fps_raw > 0.0 and not math.isnan(fps_raw)) else 30.0

        total_frames_raw = float(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0)
        total_frames = int(total_frames_raw) if total_frames_raw > 0 else 0

        width_raw = float(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0.0)
        height_raw = float(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0.0)
        width = int(width_raw) if width_raw > 0 else 0
        height = int(height_raw) if height_raw > 0 else 0

        duration_seconds_raw = float(cap.get(cv2.CAP_PROP_DURATION) or 0.0)
        duration_seconds = (
            duration_seconds_raw
            if duration_seconds_raw > 0.0 and not math.isnan(duration_seconds_raw)
            else 0.0
        )

        if duration_seconds <= 0.0:
            duration_seconds = (total_frames / fps) if (total_frames > 0 and fps > 0) else 0.0

        return {
            "video_path": video_path,
            "fps": fps,
            "total_frames": total_frames,
            "width": width,
            "height": height,
            "duration_seconds": duration_seconds,
        }
    finally:
        cap.release()


def load_video(video_path: str) -> cv2.VideoCapture:
    """Open a video file and return a `cv2.VideoCapture` handle.

    Raises:
        FileNotFoundError: If the file is missing.
        ValueError: If OpenCV cannot open the file.
    """
    if not os.path.isfile(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        cap.release()
        raise ValueError(f"Could not open video for reading: {video_path}")

    return cap


def normalize_frame(frame: Any, resize: tuple[int, int] | None = None) -> Any:
    """Validate, optionally resize, and convert a frame from BGR to RGB.

    Args:
        frame: OpenCV frame in BGR format.
        resize: Optional (width, height) target size.

    Returns:
        RGB frame as a numpy array.

    Raises:
        ValueError: If the frame is None or resize arguments are invalid.
    """
    if frame is None:
        raise ValueError("normalize_frame: received frame=None")

    processed = frame
    if resize is not None:
        if not isinstance(resize, tuple) or len(resize) != 2:
            raise ValueError(f"normalize_frame: resize must be a tuple (width, height); got {resize!r}")
        width, height = resize
        if width <= 0 or height <= 0:
            raise ValueError(f"normalize_frame: resize values must be positive; got {resize!r}")
        processed = cv2.resize(processed, (int(width), int(height)), interpolation=cv2.INTER_AREA)

    # OpenCV frames are BGR by default; convert to RGB for later processing.
    rgb_frame = cv2.cvtColor(processed, cv2.COLOR_BGR2RGB)
    return rgb_frame


def extract_frames(
    video_path: str,
    target_fps: float | None = None,
    resize: tuple[int, int] | None = None,
) -> list[dict]:
    """Extract frames from a video and return normalized RGB frames.

    Frames are returned as a list of dictionaries, each containing:
    - `frame_index`: the original frame index (0-based)
    - `timestamp`: timestamp in seconds
    - `frame_rgb`: normalized RGB frame

    Args:
        video_path: Path to the input video file.
        target_fps: Optional lower target FPS for downsampling.
        resize: Optional (width, height) resize applied before RGB conversion.

    Returns:
        List of frame dictionaries kept after optional downsampling.

    Raises:
        FileNotFoundError: If the file is missing.
        ValueError: If metadata is invalid or frame extraction terminates early.
    """
    cap = load_video(video_path)
    try:
        fps_raw = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        original_fps = fps_raw if (fps_raw > 0.0 and not math.isnan(fps_raw)) else 30.0

        total_frames_raw = float(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0)
        expected_total_frames = int(total_frames_raw) if total_frames_raw > 0 else 0

        frame_interval = 1
        if target_fps is not None:
            target_fps_val = float(target_fps)
            if not math.isnan(target_fps_val) and target_fps_val > 0.0 and target_fps_val < original_fps:
                # Sampling interval in original frames.
                frame_interval = max(1, int(original_fps / target_fps_val))

        frames: list[dict] = []
        frame_index = 0

        while True:
            ret, frame_bgr = cap.read()
            if not ret:
                # If metadata suggests more frames, treat early EOF as an error.
                if expected_total_frames > 0 and frame_index < expected_total_frames:
                    raise ValueError(
                        f"extract_frames: early termination at frame_index={frame_index}; "
                        f"expected_total_frames={expected_total_frames}"
                    )
                break

            if frame_index % frame_interval == 0:
                timestamp_sec = frame_index / original_fps if original_fps > 0 else 0.0
                frame_rgb = normalize_frame(frame_bgr, resize=resize)
                frames.append(
                    {
                        "frame_index": frame_index,
                        "timestamp": timestamp_sec,
                        "frame_rgb": frame_rgb,
                    }
                )

            frame_index += 1

        return frames
    finally:
        cap.release()
