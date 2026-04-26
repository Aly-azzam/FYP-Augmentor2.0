"""Simplified, downstream-facing SAM 2 artifact contract.

The rich ``detections.json`` / ``features.json`` documents (see
``sam2_schema.py``) are kept as internal debug artifacts. The rest of
the backend (expert reference service, learner runtime service,
inspection API, future compare flow) consumes the lightweight
``raw.json`` / ``summary.json`` contract defined here.

Rationale for a separate, flatter contract:

    * stability — the contract is the only thing downstream consumers
      should depend on; internal detection fields can evolve freely.
    * simplicity — a single ``mask_bbox_xyxy: [x1, y1, x2, y2]`` list
      is easier to consume from JS / SQL than a typed bbox object.
    * scope discipline — no trajectory / velocity / angle fields: SAM 2
      is strictly learner-local mask + ROI + centroid + area here.

All coordinates are in pixel space of the source video.
"""

from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Per-frame raw record
# ---------------------------------------------------------------------------

class SAM2RawFrame(BaseModel):
    """Per-frame SAM 2 mask summary used by the rest of the system."""

    frame_index: int = Field(..., ge=0, description="Absolute frame index in the source video.")
    timestamp_sec: Optional[float] = Field(default=None, ge=0.0)
    object: str = Field(
        default="learner_local_region",
        description="Tracked object label for this frame.",
    )

    has_mask: bool = Field(..., description="True when SAM 2 produced a non-empty mask for this frame.")
    mask_bbox_xyxy: Optional[List[int]] = Field(
        default=None,
        min_length=4,
        max_length=4,
        description="Axis-aligned bbox as [x1, y1, x2, y2] in pixels, or null.",
    )
    mask_centroid_xy: Optional[List[float]] = Field(
        default=None,
        min_length=2,
        max_length=2,
        description="[cx, cy] pixel centroid of the mask, or null.",
    )
    mask_area_px: Optional[int] = Field(
        default=None,
        ge=0,
        description="Number of foreground pixels in the mask, or null.",
    )
    quality_flag: str = Field(
        default="ok",
        description="Temporal stability status for this frame.",
    )
    status: str = Field(
        default="ok",
        description="Alias of quality_flag for learner-region status.",
    )

    # Preferred stable-region aliases (kept alongside legacy names).
    bbox: Optional[List[int]] = Field(
        default=None,
        min_length=4,
        max_length=4,
        description="Alias of mask_bbox_xyxy for stable-region consumers.",
    )
    centroid: Optional[List[float]] = Field(
        default=None,
        min_length=2,
        max_length=2,
        description="Alias of mask_centroid_xy for stable-region consumers.",
    )
    area: Optional[int] = Field(
        default=None,
        ge=0,
        description="Alias of mask_area_px for stable-region consumers.",
    )


class SAM2RawDocument(BaseModel):
    """Top-level payload written to ``raw.json``."""

    run_id: str
    source_video_path: str
    pipeline_name: str = Field(..., description="Name of the pipeline that produced this document.")
    model_name: str
    fps: float
    frame_count: int = Field(..., ge=0)
    width: int = Field(..., gt=0)
    height: int = Field(..., gt=0)
    target_object_id: int = Field(default=1)

    frames: List[SAM2RawFrame] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

class SAM2SummaryWorkingRegion(BaseModel):
    """Flat bbox the mistake/region logic can compare against."""

    x_min: int
    y_min: int
    x_max: int
    y_max: int
    width: int
    height: int
    padding_px: int = Field(default=0, ge=0)
    frame_count_used: int = Field(default=0, ge=0)


class SAM2SummaryDocument(BaseModel):
    """Top-level payload written to ``summary.json``.

    Aggregated, region-support metrics only — no trajectory / angle /
    vibration analytics. Consumers use this as the "at a glance" view of
    a SAM 2 run; per-frame detail lives in ``raw.json``.
    """

    run_id: str
    pipeline_name: str
    model_name: str

    total_frames: int = Field(..., ge=0)
    frames_with_mask: int = Field(..., ge=0)
    detection_rate: float = Field(..., ge=0.0, le=1.0)

    mean_mask_area_px: Optional[float] = None
    min_mask_area_px: Optional[int] = None
    max_mask_area_px: Optional[int] = None
    std_mask_area_px: Optional[float] = None

    mean_centroid_speed_px_per_frame: Optional[float] = Field(
        default=None,
        description=(
            "Mean inter-frame centroid displacement in pixels (region-support "
            "only, NOT trajectory scoring)."
        ),
    )

    track_fragmentation_count: int = Field(
        default=0,
        ge=0,
        description=(
            "Number of transitions from has_mask=True to has_mask=False "
            "across consecutive processed frames. Proxy for track stability."
        ),
    )

    working_region: Optional[SAM2SummaryWorkingRegion] = None

    warnings: List[str] = Field(default_factory=list)
