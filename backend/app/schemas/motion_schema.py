"""Motion Representation output contract.

Canonical units:
- coordinates: normalized
- time: seconds
- angles: degrees
- velocity: normalized units / second
"""

from typing import Optional

from pydantic import BaseModel


class FrameMotion(BaseModel):
    frame_index: int
    timestamp_sec: float
    positions: dict[str, list[float]]  # landmark_name -> [x, y, z?]
    velocity: Optional[dict[str, list[float]]] = None
    speed: Optional[dict[str, float]] = None


class TrajectoryPoint(BaseModel):
    landmark_name: str
    points: list[list[float]]  # [[x, y, z?], ...]


class ToolState(BaseModel):
    frame_index: int
    tool_class: str
    present: bool
    confidence: float
    position_normalized: Optional[list[float]] = None


class MotionOutput(BaseModel):
    """Full output contract of the Motion Representation Engine."""

    video_id: str
    fps: float
    sequence_length: int
    tracked_landmarks: list[str]
    tracked_tools: list[str]
    frames: list[FrameMotion]
    trajectories: list[TrajectoryPoint] = []
    tool_states: list[ToolState] = []
