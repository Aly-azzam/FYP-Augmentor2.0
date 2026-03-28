"""Schemas for the Evaluation Engine output.

These models define the clean JSON contract returned by the evaluation
module after comparing learner motion against expert motion.
"""

from datetime import datetime
from typing import Any, Optional
from uuid import UUID

from pydantic import BaseModel, Field


class MetricBreakdownItem(BaseModel):
    raw_value: float = Field(..., ge=0.0, le=1.0)
    normalized_quality: float = Field(..., ge=0.0, le=1.0)
    weight: float = Field(..., ge=0.0, le=1.0)
    contribution: float = Field(..., ge=0.0)


class MetricSet(BaseModel):
    angle_deviation: float = Field(..., ge=0.0)
    trajectory_deviation: float = Field(..., ge=0.0)
    velocity_difference: float = Field(..., ge=0.0)
    smoothness_score: float = Field(..., ge=0.0, le=1.0)
    timing_score: float = Field(..., ge=0.0, le=1.0)
    hand_openness_deviation: float = Field(..., ge=0.0, le=1.0)
    tool_alignment_deviation: float = Field(0.0, ge=0.0)
    dtw_similarity: float = Field(1.0, ge=0.0, le=1.0)


class EvaluationSummary(BaseModel):
    main_strength: str
    main_weakness: str
    focus_area: str


class ExplanationOutput(BaseModel):
    mode: str = "rule_based"
    explanation: str
    strengths: list[str] = Field(default_factory=list)
    weaknesses: list[str] = Field(default_factory=list)
    advice: str


class KeyErrorMoment(BaseModel):
    error_type: str
    label: str
    start_aligned_index: int = Field(..., ge=0)
    end_aligned_index: int = Field(..., ge=0)
    start_expert_frame: int = Field(..., ge=0)
    end_expert_frame: int = Field(..., ge=0)
    start_learner_frame: int = Field(..., ge=0)
    end_learner_frame: int = Field(..., ge=0)
    severity: float = Field(..., ge=0.0, le=1.0)
    semantic_phase: Optional["SemanticPhaseSegment"] = None
    semantic_label: Optional[str] = None


class SemanticPhaseSegment(BaseModel):
    label: str
    start_frame: int = Field(..., ge=0)
    end_frame: int = Field(..., ge=0)


class SemanticPhasesOut(BaseModel):
    phases: list[SemanticPhaseSegment] = Field(default_factory=list)
    motion_intensity_series: list[float] = Field(default_factory=list)


class VLMInputPayload(BaseModel):
    score: int = Field(..., ge=0, le=100)
    metrics: MetricSet
    summary: EvaluationSummary
    per_metric_breakdown: dict[str, MetricBreakdownItem] = Field(default_factory=dict)
    key_error_moments: list[KeyErrorMoment] = Field(default_factory=list)
    semantic_phases: dict[str, SemanticPhasesOut] = Field(default_factory=dict)
    explanation_payload: dict[str, Any] = Field(default_factory=dict)
    explanation: ExplanationOutput = Field(
        default_factory=lambda: ExplanationOutput(
            mode="rule_based",
            explanation="",
            strengths=[],
            weaknesses=[],
            advice="",
        )
    )


class EvaluationResult(BaseModel):
    evaluation_id: str
    score: int = Field(..., ge=0, le=100)
    metrics: MetricSet
    per_metric_breakdown: dict[str, MetricBreakdownItem] = Field(default_factory=dict)
    key_error_moments: list[KeyErrorMoment] = Field(default_factory=list)
    semantic_phases: dict[str, SemanticPhasesOut] = Field(default_factory=dict)
    explanation: ExplanationOutput = Field(
        default_factory=lambda: ExplanationOutput(
            mode="rule_based",
            explanation="",
            strengths=[],
            weaknesses=[],
            advice="",
        )
    )
    summary: EvaluationSummary
    vlm_payload: VLMInputPayload


class EvaluationMetrics(BaseModel):
    angle_deviation: float = Field(..., ge=0.0)
    trajectory_deviation: float = Field(..., ge=0.0)
    velocity_difference: float = Field(..., ge=0.0)
    smoothness_score: float = Field(..., ge=0.0, le=1.0)
    timing_score: float = Field(..., ge=0.0, le=1.0)
    hand_openness_deviation: float = Field(..., ge=0.0, le=1.0)
    tool_alignment_deviation: float = Field(0.0, ge=0.0)
    dtw_similarity: float = Field(1.0, ge=0.0, le=1.0)


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


class PersistedEvaluationOut(BaseModel):
    id: str
    attempt_id: str
    score: float
    metrics: dict[str, Any] = Field(default_factory=dict)
    per_metric_breakdown: dict[str, Any] = Field(default_factory=dict)
    key_error_moments: list[dict[str, Any]] = Field(default_factory=list)
    semantic_phases: dict[str, Any] = Field(default_factory=dict)
    explanation: dict[str, Any] = Field(default_factory=dict)
    created_at: Optional[datetime] = None


class EvaluationHistoryOut(BaseModel):
    evaluations: list[PersistedEvaluationOut] = Field(default_factory=list)


class AttemptProgressOut(BaseModel):
    attempt_id: str
    best_score: float
    average_score: float
    attempt_count: int
    latest_score: float
