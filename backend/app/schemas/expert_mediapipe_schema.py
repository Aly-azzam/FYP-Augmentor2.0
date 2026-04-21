"""Pydantic schemas for the expert MediaPipe preprocessing flow.

These schemas describe the request/response payloads used by the
one-time offline expert preprocessing pipeline implemented in
``app.services.expert_mediapipe_service``.

They intentionally mirror the existing Video / Chapter shape so the
frontend (if/when it consumes the admin endpoint) does not need a new
mental model for expert references.
"""

from __future__ import annotations

from datetime import datetime
from typing import Optional
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class ExpertMediaPipeProcessRequest(BaseModel):
    """Admin request body for expert MediaPipe preprocessing.

    Exactly one of ``expert_video_id`` or ``chapter_id`` must be provided.
    ``video_path`` is optional; when omitted, the service uses the
    ``Video.file_path`` already stored in the DB.
    """

    expert_video_id: Optional[UUID] = Field(
        default=None,
        description="ID of the expert Video row (the canonical expert reference).",
    )
    chapter_id: Optional[UUID] = Field(
        default=None,
        description="Chapter ID. The service resolves the expert video for this chapter.",
    )
    video_path: Optional[str] = Field(
        default=None,
        description=(
            "Optional override. Accepts an absolute path, a CWD-relative path, "
            "or a storage-relative key. When omitted, the expert Video.file_path "
            "stored in the DB is used."
        ),
    )
    overwrite: bool = Field(
        default=False,
        description=(
            "If true, rerun MediaPipe and replace any previously saved expert "
            "outputs. If false and the expert already has completed outputs, "
            "the existing reference is returned without reprocessing."
        ),
    )
    max_num_hands: int = Field(default=2, ge=1, le=4)
    min_detection_confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    min_tracking_confidence: float = Field(default=0.5, ge=0.0, le=1.0)


class ExpertMediaPipeSummary(BaseModel):
    """Summary fields extracted from the expert ``metadata.json``."""

    fps: Optional[float] = None
    frame_count: Optional[int] = None
    detection_rate: Optional[float] = None
    selected_hand_policy: Optional[str] = None
    width: Optional[int] = None
    height: Optional[int] = None
    total_frames: Optional[int] = None
    frames_with_detection: Optional[int] = None
    right_hand_selected_count: Optional[int] = None
    left_hand_selected_count: Optional[int] = None


class ExpertMediaPipeReference(BaseModel):
    """Response payload describing a saved expert MediaPipe reference."""

    model_config = ConfigDict(from_attributes=True)

    expert_video_id: UUID
    expert_code: str = Field(
        ...,
        description=(
            "Stable folder name used under storage/expert/mediapipe/. "
            "In the current schema this is the expert Video.id."
        ),
    )
    chapter_id: Optional[UUID] = None
    title: Optional[str] = Field(
        default=None, description="Chapter title (best-effort)."
    )

    source_path: str
    detections_path: str
    features_path: str
    metadata_path: str
    annotated_path: Optional[str] = Field(
        default=None,
        description=(
            "Stable path to the annotated expert video "
            "(storage/expert/mediapipe/{expert_code}/annotated.mp4). "
            "Null when the MediaPipe visualization step failed."
        ),
    )

    mediapipe_status: str
    mediapipe_processed_at: Optional[datetime] = None
    pipeline_version: Optional[str] = None

    summary: ExpertMediaPipeSummary = Field(default_factory=ExpertMediaPipeSummary)

    reused_existing: bool = Field(
        default=False,
        description=(
            "True when an already-completed reference was returned without "
            "rerunning the MediaPipe pipeline (overwrite=false)."
        ),
    )
