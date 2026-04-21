"""Pydantic schemas for the expert SAM 2 preprocessing flow.

Mirrors ``expert_mediapipe_schema.py``. These schemas describe the
request / response payloads used by the one-time offline expert SAM 2
preprocessing pipeline implemented in
``app.services.expert_sam2_service``.

The frontend rarely consumes this directly (the admin flow is mostly
driven from the CLI), but we keep it alongside the MediaPipe expert
schema so any admin UI we add later has a consistent shape for both
models.
"""

from __future__ import annotations

from datetime import datetime
from typing import Optional
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class ExpertSam2ProcessRequest(BaseModel):
    """Admin request body for expert SAM 2 preprocessing.

    Exactly one of ``expert_video_id`` or ``chapter_id`` must be provided.
    ``video_path`` is optional; when omitted the service uses the
    ``Video.file_path`` already stored in the DB. The MediaPipe features
    must have been computed first — SAM 2 uses them to auto-derive its
    initial prompt (no user clicks).
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
            "Optional override for the source video path. Absolute / CWD-relative / "
            "storage-relative. When omitted the expert Video.file_path is used."
        ),
    )
    overwrite: bool = Field(
        default=False,
        description=(
            "If true, rerun SAM 2 and replace any previously saved expert "
            "outputs. If false and the expert already has completed outputs, "
            "the existing reference is returned without reprocessing."
        ),
    )
    render_annotation: bool = Field(
        default=True,
        description="When false, skip writing annotated.mp4 (JSON artifacts are always written).",
    )


class ExpertSam2Summary(BaseModel):
    """Summary fields extracted from the expert ``summary.json`` + ``metadata.json``."""

    fps: Optional[float] = None
    frame_count: Optional[int] = None
    width: Optional[int] = None
    height: Optional[int] = None
    model_name: Optional[str] = None

    total_frames: Optional[int] = None
    frames_with_mask: Optional[int] = None
    detection_rate: Optional[float] = None

    mean_mask_area_px: Optional[float] = None
    mean_centroid_speed_px_per_frame: Optional[float] = None
    track_fragmentation_count: Optional[int] = None


class ExpertSam2Reference(BaseModel):
    """Response payload describing a saved expert SAM 2 reference."""

    model_config = ConfigDict(from_attributes=True)

    expert_video_id: UUID
    expert_code: str = Field(
        ...,
        description=(
            "Stable folder name used under storage/expert/sam2/. "
            "In the current schema this is the expert Video.id."
        ),
    )
    chapter_id: Optional[UUID] = None
    title: Optional[str] = Field(
        default=None, description="Chapter title (best-effort)."
    )

    source_path: str
    raw_path: str
    summary_path: str
    metadata_path: str
    annotated_path: Optional[str] = Field(
        default=None,
        description=(
            "Stable path to the annotated expert video "
            "(storage/expert/sam2/{expert_code}/annotated.mp4). "
            "Null when SAM 2 visualization failed or was skipped."
        ),
    )
    init_debug_image_path: Optional[str] = Field(
        default=None,
        description=(
            "Stable path to the init-prompt debug image (point/box overlay "
            "on the selected initialization frame). Useful for diffing expert "
            "vs learner prompt placement side by side."
        ),
    )

    sam2_status: str
    sam2_processed_at: Optional[datetime] = None
    pipeline_version: Optional[str] = None

    summary: ExpertSam2Summary = Field(default_factory=ExpertSam2Summary)

    reused_existing: bool = Field(
        default=False,
        description=(
            "True when an already-completed reference was returned without "
            "rerunning the SAM 2 pipeline (overwrite=false)."
        ),
    )
