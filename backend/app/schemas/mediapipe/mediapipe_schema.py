"""Pydantic schemas for the MediaPipe Hands extraction pipeline.

These schemas are the source of truth for the three JSON documents that a
single extraction run produces:

    backend/storage/mediapipe/runs/<run_id>/detections.json   -> MediaPipeDetectionsDocument
    backend/storage/mediapipe/runs/<run_id>/metadata.json     -> MediaPipeRunMeta
    backend/storage/mediapipe/runs/<run_id>/features.json     -> MediaPipeFeaturesDocument

All coordinates are stored in MediaPipe's normalized image space where
``x`` and ``y`` are in [0, 1] (relative to image width / height) and ``z``
is a depth estimate relative to the wrist.
"""

from __future__ import annotations

from typing import Dict, List, Literal, Optional

from pydantic import BaseModel, Field


HandednessLiteral = Literal["Left", "Right", "Unknown"]


class MediaPipeLandmarkPoint(BaseModel):
    """One of the 21 MediaPipe hand landmarks for a single frame."""

    index: int = Field(..., ge=0, le=20, description="MediaPipe landmark index (0..20).")
    name: str = Field(..., description="Human readable name, e.g. 'wrist', 'index_finger_tip'.")
    x: float = Field(..., description="Normalized x in [0, 1] relative to image width.")
    y: float = Field(..., description="Normalized y in [0, 1] relative to image height.")
    z: float = Field(..., description="Depth estimate relative to wrist (MediaPipe convention).")


class MediaPipeSelectedHandRaw(BaseModel):
    """Raw data for one hand detected on a given frame."""

    handedness: HandednessLiteral
    handedness_score: float = Field(
        ..., ge=0.0, le=1.0, description="MediaPipe handedness classification score."
    )
    detection_confidence: float = Field(
        ..., ge=0.0, le=1.0, description="Detection / presence confidence reported by MediaPipe."
    )

    wrist: List[float] = Field(
        ..., min_length=3, max_length=3, description="[x, y, z] of landmark 0 (wrist)."
    )
    index_tip: List[float] = Field(
        ..., min_length=3, max_length=3, description="[x, y, z] of landmark 8 (index finger tip)."
    )
    thumb_tip: List[float] = Field(
        ..., min_length=3, max_length=3, description="[x, y, z] of landmark 4 (thumb tip)."
    )
    middle_tip: List[float] = Field(
        ..., min_length=3, max_length=3, description="[x, y, z] of landmark 12 (middle finger tip)."
    )
    hand_center: List[float] = Field(
        ..., min_length=3, max_length=3, description="[x, y, z] palm center (mean of palm landmarks)."
    )

    landmarks: List[MediaPipeLandmarkPoint] = Field(
        ..., min_length=21, max_length=21, description="All 21 MediaPipe hand landmarks."
    )


class MediaPipeFrameRaw(BaseModel):
    """Raw per-frame record written to detections.json."""

    frame_index: int = Field(..., ge=0)
    timestamp_sec: float = Field(..., ge=0.0)
    has_detection: bool
    num_hands_detected: int = Field(..., ge=0)
    hands: List[MediaPipeSelectedHandRaw] = Field(default_factory=list)
    selected_hand: Optional[MediaPipeSelectedHandRaw] = None


class MediaPipeDetectionsDocument(BaseModel):
    """Top level document serialized to detections.json."""

    run_id: str
    source_video_path: str
    fps: float
    frame_count: int = Field(..., ge=0)
    width: int = Field(..., gt=0)
    height: int = Field(..., gt=0)
    selected_hand_policy: str
    frames: List[MediaPipeFrameRaw] = Field(default_factory=list)


class MediaPipeRunMeta(BaseModel):
    """Top level document serialized to metadata.json."""

    run_id: str
    source_video_path: str
    fps: float
    frame_count: int = Field(..., ge=0)
    width: int = Field(..., gt=0)
    height: int = Field(..., gt=0)
    created_at: str = Field(..., description="ISO-8601 UTC timestamp when the run was produced.")
    selected_hand_policy: str

    total_frames: int = Field(..., ge=0)
    frames_with_detection: int = Field(..., ge=0)
    detection_rate: float = Field(..., ge=0.0, le=1.0)
    right_hand_selected_count: int = Field(..., ge=0)
    left_hand_selected_count: int = Field(..., ge=0)


class MediaPipeHandBoundingBox(BaseModel):
    """Axis-aligned bounding box for the selected hand."""

    x_min: float
    y_min: float
    x_max: float
    y_max: float
    width: float
    height: float


class MediaPipeJointAngles(BaseModel):
    """Selected joint angles (in degrees) for the selected hand.

    These are the angles we want to visualize / compare across experts and
    learners. They are intentionally a small, interpretable subset.
    """

    wrist_index_mcp_index_tip: float
    wrist_middle_mcp_middle_tip: float
    wrist_ring_mcp_ring_tip: float
    index_mcp_middle_mcp_pinky_mcp: float
    thumb_mcp_thumb_ip_thumb_tip: float


class MediaPipeHandFrameFeatures(BaseModel):
    """Per-frame derived features for one handedness track."""

    handedness: HandednessLiteral
    detected: bool = Field(..., description="True when MediaPipe detected this hand on this frame.")
    fallback_used: bool = Field(
        default=False,
        description="True when values were carried forward from the last good detection.",
    )
    confidence: float = Field(
        ..., ge=0.0, le=1.0, description="Detection / fallback confidence for this hand."
    )

    wrist: List[float] = Field(..., min_length=3, max_length=3)
    index_tip: List[float] = Field(..., min_length=3, max_length=3)
    thumb_tip: List[float] = Field(..., min_length=3, max_length=3)
    middle_tip: List[float] = Field(..., min_length=3, max_length=3)
    hand_center: List[float] = Field(..., min_length=3, max_length=3)
    landmarks: List[List[float]] = Field(
        ..., min_length=21, max_length=21, description="All 21 landmarks as [x, y, z]."
    )

    hand_bbox: MediaPipeHandBoundingBox
    wrist_angle_deg: float = Field(
        ..., description="Alias for hand_orientation_deg; angle of wrist -> middle MCP."
    )
    hand_orientation_deg: float
    wrist_relative_landmarks: Dict[str, List[float]]
    joint_angles: MediaPipeJointAngles
    wrist_velocity: Optional[List[float]] = Field(default=None, min_length=3, max_length=3)
    index_tip_velocity: Optional[List[float]] = Field(default=None, min_length=3, max_length=3)
    thumb_tip_velocity: Optional[List[float]] = Field(default=None, min_length=3, max_length=3)
    middle_tip_velocity: Optional[List[float]] = Field(default=None, min_length=3, max_length=3)
    trajectory_history: List[List[float]] = Field(default_factory=list)


class MediaPipeFrameFeatures(BaseModel):
    """Per-frame derived features written to features.json."""

    frame_index: int = Field(..., ge=0)
    timestamp_sec: float = Field(..., ge=0.0)
    has_detection: bool

    handedness: Optional[HandednessLiteral] = None
    hands: List[MediaPipeHandFrameFeatures] = Field(default_factory=list)
    left_hand_features: Optional[MediaPipeHandFrameFeatures] = None
    right_hand_features: Optional[MediaPipeHandFrameFeatures] = None

    wrist: Optional[List[float]] = Field(
        default=None, description="[x, y, z] of wrist when detected."
    )
    hand_center: Optional[List[float]] = Field(
        default=None, description="[x, y, z] palm center when detected."
    )

    hand_bbox: Optional[MediaPipeHandBoundingBox] = None
    hand_orientation_deg: Optional[float] = Field(
        default=None,
        description="Angle in degrees from wrist -> middle finger MCP "
        "measured in image space (0 = +x axis, CCW).",
    )

    wrist_relative_landmarks: Optional[Dict[str, List[float]]] = Field(
        default=None,
        description="All 21 landmarks expressed as [x, y, z] relative to the wrist.",
    )

    joint_angles: Optional[MediaPipeJointAngles] = None

    # Velocities are SUPPORT metrics only, not main scoring signals.
    wrist_velocity: Optional[List[float]] = Field(
        default=None, min_length=3, max_length=3
    )
    index_tip_velocity: Optional[List[float]] = Field(
        default=None, min_length=3, max_length=3
    )
    thumb_tip_velocity: Optional[List[float]] = Field(
        default=None, min_length=3, max_length=3
    )
    middle_tip_velocity: Optional[List[float]] = Field(
        default=None, min_length=3, max_length=3
    )

    trajectory_history: List[List[float]] = Field(
        default_factory=list,
        description="Wrist trajectory up to this frame (last N entries, each [x, y]).",
    )


class MediaPipeFeaturesDocument(BaseModel):
    """Top level document serialized to features.json."""

    run_id: str
    source_video_path: str
    fps: float
    frame_count: int = Field(..., ge=0)
    trajectory_history_size: int = Field(
        ..., ge=1, description="Maximum length of the wrist trajectory history per frame."
    )
    frames: List[MediaPipeFrameFeatures] = Field(default_factory=list)
