"""Motion Representation output contract.

Canonical units:
- coordinates: normalized
- time: seconds
- angles: degrees
- velocity: normalized units / second
"""

from typing import Optional

from pydantic import BaseModel


class MotionFrame(BaseModel):
    frame_index: int
    timestamp: float
    positions: dict[str, list[float]]


class MotionSequenceOutput(BaseModel):
    """Step 1 output contract for cleaned time-ordered motion sequences."""

    video_id: str
    fps: float
    sequence_length: int
    tracked_landmarks: list[str]
    frames: list[MotionFrame]


class EnrichedMotionFrame(MotionFrame):
    velocity: dict[str, list[float]]
    speed: dict[str, float]
    acceleration: dict[str, list[float]]
    trajectory_point: dict[str, list[float]]


class MotionRepresentationOutput(BaseModel):
    """Step 4 output contract for enriched motion representation."""

    video_id: str
    fps: float
    sequence_length: int
    tracked_landmarks: list[str]
    frames: list[EnrichedMotionFrame]


# Backward-compatible alias used by existing motion stubs.
MotionOutput = MotionRepresentationOutput
