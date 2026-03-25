"""Motion Representation output contracts.

Canonical units:
- coordinates: normalized
- time: seconds
- angles: degrees
- velocity: normalized units / second
- acceleration: normalized units / second^2
"""

from typing import Optional

from pydantic import BaseModel, Field


class MotionFrame(BaseModel):
    """Legacy motion-frame contract kept for backward compatibility."""

    frame_index: int
    timestamp: float
    positions: dict[str, list[float]]


class MotionSequenceOutput(BaseModel):
    """Legacy motion-sequence contract kept for backward compatibility."""

    video_id: str
    fps: float
    sequence_length: int
    tracked_landmarks: list[str]
    frames: list[MotionFrame]


class EnrichedMotionFrame(MotionFrame):
    """Legacy enriched motion-frame contract kept for backward compatibility."""

    velocity: dict[str, list[float]]
    speed: dict[str, float]
    acceleration: dict[str, list[float]]
    trajectory_point: dict[str, list[float]]


class HandMotionFeatures(BaseModel):
    present: bool
    wrist_position: list[float]
    palm_center: list[float]
    hand_scale: float
    wrist_relative_to_palm: list[float]
    fingertip_positions: dict[str, list[float]]
    fingertip_relative_positions: dict[str, list[float]]
    relative_distances: dict[str, float]
    joint_angles: dict[str, float]
    hand_openness: float
    pinch_distance: float
    finger_spread: float
    wrist_velocity: list[float]
    palm_velocity: list[float]
    wrist_acceleration: list[float]
    palm_acceleration: list[float]
    flattened_feature_vector: list[float]


class MotionFrameFeatures(BaseModel):
    frame_index: int
    timestamp_sec: float
    left_hand_features: HandMotionFeatures
    right_hand_features: HandMotionFeatures
    flattened_feature_vector: list[float]


class MotionRepresentationOutput(BaseModel):
    """Step 2.4 output contract for comparison-ready motion features."""

    video_id: str
    fps: float
    total_frames: int
    sequence_length: int
    frames: list[MotionFrameFeatures]
    video_path: Optional[str] = None
    coordinate_system: str = "normalized"
    hand_feature_vector_dim: int = Field(..., ge=1)
    frame_feature_vector_dim: int = Field(..., ge=1)


# Backward-compatible alias used by existing motion stubs and newer pipeline code.
MotionOutput = MotionRepresentationOutput
