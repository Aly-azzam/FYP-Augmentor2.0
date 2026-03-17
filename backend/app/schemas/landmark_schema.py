"""Perception output contract.

Canonical units:
- coordinates: normalized (0..1) MediaPipe-style
- timestamps: seconds
"""

from typing import Optional

from pydantic import BaseModel


class LandmarkPoint(BaseModel):
    x: float
    y: float
    z: Optional[float] = None
    confidence: Optional[float] = None


class ToolDetection(BaseModel):
    tool_class: str
    confidence: float
    bbox_normalized: Optional[list[float]] = None  # [x_min, y_min, x_max, y_max]


class FrameLandmarks(BaseModel):
    frame_index: int
    timestamp_sec: float
    left_hand: Optional[list[LandmarkPoint]] = None
    right_hand: Optional[list[LandmarkPoint]] = None
    tool_detections: list[ToolDetection] = []


class PerceptionOutput(BaseModel):
    """Full output contract of the Perception Engine."""

    video_id: str
    fps: float
    total_frames: int
    frames: list[FrameLandmarks]
    vjepa_features: Optional[dict] = None  # placeholder for V-JEPA integration
