"""Sequence Service — build motion sequences from raw landmarks.

Responsibilities:
- clean and filter landmark sequences
- handle missing frames / interpolation
- produce ordered tracked landmark sequences

Owner: Motion Representation Engine (Person 2)
"""

from app.schemas.landmark_schema import PerceptionOutput


async def build_sequences(perception_output: PerceptionOutput) -> dict:
    """Build cleaned motion sequences from perception landmarks.

    Returns dict with tracked_landmarks and per-frame position arrays.
    """
    # TODO: implement sequence construction with gap interpolation
    return {
        "tracked_landmarks": [],
        "frames": [],
        "sequence_length": 0,
    }
