"""MediaPipe Service — hand landmark extraction.

Responsibilities:
- run MediaPipe Hands on video frames
- output per-frame normalized hand landmarks

Owner: Perception Engine (Person 1)
"""

from pathlib import Path

from app.schemas.landmark_schema import PerceptionOutput


async def extract_hand_landmarks(video_path: Path) -> list[dict]:
    """Extract hand landmarks from all video frames.

    Returns list of per-frame landmark dicts with normalized coordinates.
    """
    # TODO: implement MediaPipe Hands extraction
    return []


async def run_mediapipe_pipeline(video_path: Path, fps: float, total_frames: int) -> dict:
    """Run full MediaPipe extraction and return structured data.

    Returns dict compatible with PerceptionOutput.frames field.
    """
    # TODO: implement full extraction loop
    return {
        "frames": [],
        "usable_frame_count": 0,
    }
