"""Perception Engine: perception pipeline orchestration.

This module orchestrates the end-to-end perception flow:
`video -> frames -> hand landmarks` by connecting:
- preprocessing_service: frame extraction (and RGB normalization)
- mediapipe_service: MediaPipe Hands landmark extraction (left/right)
"""

from typing import List, Dict

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

        # Avoid silent downstream failures: ensure core values are present.
        if frame["frame_rgb"] is None:
            raise ValueError(f"validate_frames_structure: frames[{i}].frame_rgb must not be None.")


def run_perception_pipeline(video_path: str) -> Dict:
    """Run the perception pipeline end-to-end (stub-free orchestration).

    Steps:
    1) Extract frames from the input video using `extract_frames`.
    2) Initialize the MediaPipe Hands detector.
    3) Process extracted frames into left/right hand landmarks using
       `process_video_frames`.
    4) Validate output lengths and frame list shapes.
    5) Return a final structured dict (no persistence or metrics yet).
    """
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
        "video_path": video_path,
        "total_frames": len(frames),
        "processed_frames": len(processed_frames),
        "frames": processed_frames,
    }
