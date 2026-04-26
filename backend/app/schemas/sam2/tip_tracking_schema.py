"""Learner-side scissor-tip tracking schemas.

SAM2 remains responsible for a stable active hand/tool ROI. This schema
captures the second stage that tracks a manually-seeded scissor tip
inside that ROI across frames.
"""

from __future__ import annotations

from typing import List, Literal, Optional

from pydantic import BaseModel, Field


class SAM2TipInitialization(BaseModel):
    frame_index: int = Field(..., ge=0)
    tip_point: List[float] = Field(..., min_length=2, max_length=2)
    object: Literal["scissor_tip"] = "scissor_tip"
    source: Literal["manual_click"] = "manual_click"


class SAM2TipTrackingFrame(BaseModel):
    frame_index: int = Field(..., ge=0)
    tip_point: List[float] = Field(..., min_length=2, max_length=2)
    tip_bbox: Optional[List[int]] = Field(default=None, min_length=4, max_length=4)
    tracking_status: Literal["ok", "unstable", "reused_previous", "missing"] = "ok"
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    inside_sam_roi: bool = True


class SAM2TipTrackingDocument(BaseModel):
    run_id: str
    source_video_path: str
    object: Literal["scissor_tip"] = "scissor_tip"
    initialization: SAM2TipInitialization
    frames: List[SAM2TipTrackingFrame] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
