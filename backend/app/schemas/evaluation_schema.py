"""Schemas for the Evaluation Engine output.

These models define the clean JSON contract returned by the evaluation
module after comparing learner motion against expert motion.
"""

from datetime import datetime
from typing import Optional
from uuid import UUID

from pydantic import BaseModel, Field


class MetricSet(BaseModel):
    angle_deviation: float = Field(..., ge=0.0)
    trajectory_deviation: float = Field(..., ge=0.0)
    velocity_difference: float = Field(..., ge=0.0)
    tool_alignment_deviation: float = Field(..., ge=0.0)


class EvaluationSummary(BaseModel):
    main_strength: str
    main_weakness: str
    focus_area: str


class VLMInputPayload(BaseModel):
    score: int = Field(..., ge=0, le=100)
    metrics: MetricSet
    summary: EvaluationSummary


class EvaluationResult(BaseModel):
    evaluation_id: str
    score: int = Field(..., ge=0, le=100)
    metrics: MetricSet
    summary: EvaluationSummary
    vlm_payload: VLMInputPayload


class EvaluationMetrics(BaseModel):
    angle_deviation: float = Field(..., ge=0.0)
    trajectory_deviation: float = Field(..., ge=0.0)
    velocity_difference: float = Field(..., ge=0.0)
    tool_alignment_deviation: float = Field(..., ge=0.0)


class EvaluationResultOut(BaseModel):
    evaluation_id: UUID
    chapter_id: UUID
    expert_video_id: UUID
    learner_video_id: UUID
    status: str
    score: int = Field(..., ge=0, le=100)
    metrics: EvaluationMetrics
    summary: Optional[str] = None
    ai_text: Optional[str] = None
    comparison_media: Optional[dict] = None
    warnings: list[dict] = Field(default_factory=list)
    pipeline_version: Optional[str] = None
    model_version: Optional[str] = None
    config_version: Optional[str] = None
    created_at: Optional[datetime] = None


class HistoryEntry(BaseModel):
    attempt_id: UUID
    chapter_id: UUID
    course_id: UUID
    course_title: str
    chapter_title: str
    status: str
    score: Optional[int] = None
    created_at: datetime


class ProgressOut(BaseModel):
    course_id: UUID
    course_title: str
    completed_chapters: int
    total_chapters: int
    completion_ratio: float
