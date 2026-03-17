"""Perception Service — orchestrates the full perception stage.

Combines:
- preprocessing / validation
- MediaPipe hand landmark extraction
- tool detection
- optional V-JEPA features

Produces: PerceptionOutput contract

Owner: Perception Engine (Person 1)
"""

from pathlib import Path

from app.schemas.landmark_schema import PerceptionOutput


async def run_perception(video_path: Path, video_id: str) -> PerceptionOutput:
    """Run full perception pipeline on a single video.

    Steps:
    1. Extract metadata (fps, frames)
    2. Run MediaPipe hand landmarks
    3. Run tool detection
    4. Optionally extract V-JEPA features
    5. Assemble PerceptionOutput

    Returns PerceptionOutput schema.
    """
    # TODO: wire real sub-services
    return PerceptionOutput(
        video_id=video_id,
        fps=30.0,
        total_frames=450,
        frames=[],
        vjepa_features=None,
    )
