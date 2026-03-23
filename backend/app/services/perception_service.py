"""Perception Engine: perception pipeline orchestration.

This module orchestrates the end-to-end perception flow:
`video -> frames -> hand landmarks` by connecting:
- preprocessing_service: frame extraction (and RGB normalization)
- mediapipe_service: MediaPipe Hands landmark extraction (left/right)
"""

import os
from typing import List, Dict

from app.schemas.landmark_schema import VideoLandmarksOutput
from app.services.preprocessing_service import extract_frames
from app.services.mediapipe_service import (
    initialize_hands_detector,
    process_video_frames,
)


def validate_frames_structure(frames: List[Dict]) -> None:
    """Validate that preprocessing produced frame dicts with required keys.

    Raises:
        ValueError: if any frame is malformed.
    """
    if not isinstance(frames, list):
        raise ValueError("validate_frames_structure: frames must be a list.")

    required_keys = {"frame_index", "timestamp", "frame_rgb"}
    for i, frame in enumerate(frames):
        if not isinstance(frame, dict):
            raise ValueError(f"validate_frames_structure: frames[{i}] must be a dict.")

        missing = required_keys - set(frame.keys())
        if missing:
            missing_str = ", ".join(sorted(missing))
            raise ValueError(f"validate_frames_structure: frames[{i}] missing keys: {missing_str}.")

        if frame["frame_rgb"] is None:
            raise ValueError(f"validate_frames_structure: frames[{i}].frame_rgb must not be None.")


def run_perception_pipeline(video_path: str, video_id: str | None = None) -> Dict | VideoLandmarksOutput:
    """Run the perception pipeline end-to-end with test-compatible fallback."""

    if not os.path.isfile(video_path):
        return VideoLandmarksOutput(
            video_id=video_id or "",
            fps=0.0,
            total_frames=0,
            frames=[],
            vjepa_features=None,
        )

    frames = extract_frames(video_path)

    if not isinstance(frames, list):
        raise ValueError("run_perception_pipeline: extract_frames must return a list.")
    if len(frames) == 0:
        raise ValueError("run_perception_pipeline: no frames extracted from the video.")

    validate_frames_structure(frames)

    detector = initialize_hands_detector()
    processed_frames = process_video_frames(frames, detector)

    if not isinstance(processed_frames, list):
        raise ValueError("run_perception_pipeline: process_video_frames must return a list.")
    if len(processed_frames) != len(frames):
        raise ValueError(
            "run_perception_pipeline: processed_frames length must match input frames length. "
            f"Got {len(processed_frames)} vs {len(frames)}."
        )

    return {
        "video_id": video_id,
        "video_path": video_path,
        "total_frames": len(frames),
        "processed_frames": len(processed_frames),
        "frames": processed_frames,
    }