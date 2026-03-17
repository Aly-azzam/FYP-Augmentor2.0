"""Evaluation output contract."""

from uuid import UUID
from typing import Optional
from datetime import datetime

from pydantic import BaseModel


class Warning(BaseModel):
    code: str
    message: str


class EvaluationMetrics(BaseModel):
    angle_deviation: float
    trajectory_deviation: float
    velocity_difference: float
    tool_alignment_deviation: float


class EvaluationResultOut(BaseModel):
    evaluation_id: UUID
    chapter_id: UUID
    expert_video_id: UUID
    learner_video_id: UUID
    status: str
    score: int
    metrics: EvaluationMetrics
    summary: Optional[str] = None
    ai_text: Optional[str] = None
    comparison_media: Optional[dict] = None
    warnings: list[Warning] = []

    pipeline_version: Optional[str] = None
    model_version: Optional[str] = None
    config_version: Optional[str] = None

    created_at: Optional[datetime] = None

    model_config = {"from_attributes": True}


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
