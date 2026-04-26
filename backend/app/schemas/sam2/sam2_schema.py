"""Pydantic schemas for the SAM 2 backend pipeline.

These schemas describe the three JSON documents that a single SAM 2 run
produces (mirroring the MediaPipe layout so the two pipelines are easy
to reason about side by side):

    storage/sam2/runs/<run_id>/detections.json -> SAM2DetectionsDocument
    storage/sam2/runs/<run_id>/metadata.json   -> SAM2RunMeta
    storage/sam2/runs/<run_id>/features.json   -> SAM2FeaturesDocument

Scope restrictions (explicitly out of scope for SAM 2):

    * No trajectory scoring
    * No joint angles
    * No vibration / motion analytics

SAM 2 is strictly limited to:

    * learner-local region mask
    * ROI bounding box
    * mask centroid
    * mask area
    * derived "working region" (aggregate ROI)
    * region-level support signals used by the mistake logic later

All bbox / centroid coordinates are in pixel space (not normalized)
because SAM 2 already operates directly on pixel-space frames.
"""

from __future__ import annotations

from typing import List, Literal, Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Initializer prompts
# ---------------------------------------------------------------------------

PromptType = Literal["point", "box"]
PromptSource = Literal["mediapipe", "manual"]
SAM2ObjectType = Literal[
    "active_hand_region",
    "hand_tool_region",
    "learner_local_region",
    "sam_roi_local_region",
]


class SAM2InitPoint(BaseModel):
    """Positive-click point prompt (pixel coordinates)."""

    x: float = Field(..., description="Point x in pixels.")
    y: float = Field(..., description="Point y in pixels.")
    label: int = Field(
        default=1,
        description="SAM 2 label: 1 = positive (inside), 0 = negative (outside).",
    )


class SAM2InitBox(BaseModel):
    """Axis-aligned bounding-box prompt (pixel coordinates)."""

    x_min: float
    y_min: float
    x_max: float
    y_max: float


class SAM2InitPrompt(BaseModel):
    """Prompt fed into SAM 2 to initialize the target object.

    The backend only ever builds this prompt automatically from the
    MediaPipe output (``source="mediapipe"``). The ``"manual"`` variant
    exists for debugging / offline experiments, never for the real
    learner pipeline.
    """

    type: PromptType = Field(..., description="Prompt shape fed into SAM 2.")
    source: PromptSource = Field(
        default="mediapipe",
        description="Where the prompt came from (automated vs debug override).",
    )
    frame_index: int = Field(
        ...,
        ge=0,
        description="Absolute frame index at which the prompt is applied.",
    )
    target_object_id: int = Field(
        default=1,
        description="SAM 2 object id to attach to this prompt.",
    )
    object_type: SAM2ObjectType = Field(
        default="learner_local_region",
        description="Semantic object represented by this prompt.",
    )
    point: Optional[SAM2InitPoint] = None
    points: Optional[List[SAM2InitPoint]] = Field(
        default=None,
        description=(
            "Optional multi-point prompt list (first point should be positive, "
            "later points can be negative disambiguation cues)."
        ),
    )
    box: Optional[SAM2InitBox] = None


# ---------------------------------------------------------------------------
# Per-frame artifacts
# ---------------------------------------------------------------------------

class SAM2BoundingBox(BaseModel):
    """Axis-aligned bbox around the detected mask (pixel coordinates)."""

    x_min: int
    y_min: int
    x_max: int
    y_max: int
    width: int
    height: int


class SAM2FrameMask(BaseModel):
    """Raw per-frame SAM 2 record (detections.json).

    ``mask_path`` is relative to STORAGE_ROOT when written by the service
    so it's portable across machines; callers that need an absolute path
    should resolve it against ``settings.STORAGE_ROOT``.
    """

    frame_index: int = Field(..., ge=0, description="Absolute frame index in the source video.")
    local_frame_index: Optional[int] = Field(
        default=None,
        description="Index inside the analysis window (0-based, post-stride).",
    )
    timestamp_sec: Optional[float] = Field(default=None, ge=0.0)

    has_mask: bool = Field(..., description="True when SAM 2 produced a non-empty mask.")
    mask_area_px: Optional[int] = Field(
        default=None, ge=0, description="Number of foreground pixels in the mask."
    )
    mask_centroid_xy: Optional[List[float]] = Field(
        default=None,
        min_length=2,
        max_length=2,
        description="[x, y] pixel centroid of the mask.",
    )
    mask_bbox: Optional[SAM2BoundingBox] = None

    mask_path: Optional[str] = Field(
        default=None,
        description="Storage-relative path to the saved binary mask PNG (if any).",
    )

    temporal_source: Literal["prompted", "propagated_forward", "propagated_backward", "none"] = Field(
        default="none",
        description=(
            "Where the mask for this frame came from: the user prompt, "
            "forward propagation, backward propagation, or none."
        ),
    )
    quality_flag: Literal["ok", "unstable", "reused_previous", "missing"] = Field(
        default="ok",
        description="Lightweight temporal-stability quality marker for this frame.",
    )
    status: Literal["ok", "unstable", "reused_previous", "missing"] = Field(
        default="ok",
        description="Alias of quality_flag for learner-region consumers.",
    )


class SAM2DetectionsDocument(BaseModel):
    """Top-level payload for ``detections.json``."""

    run_id: str
    source_video_path: str
    fps: float
    frame_count: int = Field(..., ge=0)
    width: int = Field(..., gt=0)
    height: int = Field(..., gt=0)

    target_object_id: int = Field(
        default=1, description="SAM 2 object id that was tracked in this run."
    )
    init_prompt: SAM2InitPrompt = Field(
        ..., description="Prompt used to initialize the tracker."
    )

    frames: List[SAM2FrameMask] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Run metadata
# ---------------------------------------------------------------------------

class SAM2RunMeta(BaseModel):
    """Top-level payload for ``metadata.json``."""

    run_id: str
    source_video_path: str
    pipeline_name: str
    model_name: str
    device: str = Field(..., description="Compute device used, e.g. 'cuda' or 'cpu'.")
    gpu_name: Optional[str] = Field(
        default=None,
        description="Detected GPU name when running on CUDA (if available).",
    )
    model_checkpoint_path: Optional[str] = Field(
        default=None,
        description="Resolved checkpoint path used by the run.",
    )
    model_config_path: Optional[str] = Field(
        default=None,
        description="Resolved model config path used by the run.",
    )
    created_at: str = Field(..., description="ISO-8601 UTC timestamp.")

    fps: float
    frame_count: int = Field(..., ge=0)
    width: int = Field(..., gt=0)
    height: int = Field(..., gt=0)

    analysis_start_frame_index: int = Field(..., ge=0)
    analysis_end_frame_index: int = Field(..., ge=0)
    frame_stride: int = Field(..., ge=1)

    total_frames_processed: int = Field(..., ge=0)
    frames_with_mask: int = Field(..., ge=0)
    detection_rate: float = Field(..., ge=0.0, le=1.0)

    target_object_id: int = Field(default=1)
    init_prompt: SAM2InitPrompt
    warnings: List[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Derived features (centroid trail, area trajectory, working region)
# ---------------------------------------------------------------------------

class SAM2AreaStats(BaseModel):
    """Simple descriptive stats over the per-frame mask areas."""

    min_px: Optional[int] = None
    max_px: Optional[int] = None
    mean_px: Optional[float] = None
    std_px: Optional[float] = None


class SAM2WorkingRegion(BaseModel):
    """Aggregated working region derived from per-frame mask bboxes.

    This is the rectangle the mistake/region support logic can use to
    decide whether an action happens "in the right area". We intentionally
    do NOT turn this into a hard classifier here — it's a coarse ROI.
    """

    x_min: int
    y_min: int
    x_max: int
    y_max: int
    width: int
    height: int
    source: Literal["union", "quantile"] = Field(
        default="quantile",
        description="How the aggregate bbox was computed from frame bboxes.",
    )
    quantile: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="When source='quantile', the quantile used on frame bboxes.",
    )
    padding_px: int = Field(
        default=0, ge=0, description="Padding applied around the aggregate bbox."
    )
    frame_count_used: int = Field(
        default=0, ge=0, description="Number of frames that contributed a bbox."
    )


class SAM2FrameFeatures(BaseModel):
    """Per-frame derived features written to ``features.json``."""

    frame_index: int = Field(..., ge=0)
    timestamp_sec: Optional[float] = Field(default=None, ge=0.0)
    has_mask: bool

    mask_area_px: Optional[int] = Field(default=None, ge=0)
    mask_centroid_xy: Optional[List[float]] = Field(
        default=None, min_length=2, max_length=2
    )
    mask_bbox: Optional[SAM2BoundingBox] = None

    # NOTE: centroid *velocity* is intentionally captured here as a
    # region-support signal only (e.g. "did the hand move in this frame").
    # It is NOT used for trajectory scoring.
    centroid_velocity_px_per_sec: Optional[List[float]] = Field(
        default=None,
        min_length=2,
        max_length=2,
        description="Centroid velocity in pixels/sec, region-support only.",
    )


class SAM2FeaturesDocument(BaseModel):
    """Top-level payload for ``features.json``."""

    run_id: str
    source_video_path: str
    fps: float
    frame_count: int = Field(..., ge=0)
    width: int = Field(..., gt=0)
    height: int = Field(..., gt=0)
    target_object_id: int = Field(default=1)

    working_region: Optional[SAM2WorkingRegion] = Field(
        default=None,
        description=(
            "Aggregated hand working region. None when no frame produced a mask."
        ),
    )
    area_stats: SAM2AreaStats = Field(default_factory=SAM2AreaStats)

    frames: List[SAM2FrameFeatures] = Field(default_factory=list)
