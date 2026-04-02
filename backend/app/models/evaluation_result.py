import uuid
from datetime import datetime
from typing import Any, Optional

from sqlalchemy import Boolean, DateTime, ForeignKey, Numeric, String, Text, func
from sqlalchemy.dialects.postgresql import JSONB, UUID as PG_UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.orm import synonym

from app.core.database import Base


class EvaluationResult(Base):
    __tablename__ = "evaluations"

    id: Mapped[str] = mapped_column(
        PG_UUID(as_uuid=False),
        primary_key=True,
        default=lambda: str(uuid.uuid4()),
    )
    attempt_id: Mapped[str] = mapped_column(
        PG_UUID(as_uuid=False),
        ForeignKey("attempts.id"),
        nullable=False,
        index=True,
    )
    expert_video_id: Mapped[str] = mapped_column(
        PG_UUID(as_uuid=False),
        ForeignKey("videos.id"),
        nullable=False,
    )
    learner_video_id: Mapped[str] = mapped_column(
        PG_UUID(as_uuid=False),
        ForeignKey("videos.id"),
        nullable=False,
    )
    overall_score: Mapped[Optional[float]] = mapped_column(Numeric, nullable=True)
    status: Mapped[str] = mapped_column(String(50), nullable=False)
    context_confidence: Mapped[Optional[float]] = mapped_column(Numeric, nullable=True)
    score_confidence: Mapped[Optional[float]] = mapped_column(Numeric, nullable=True)
    explanation_confidence: Mapped[Optional[float]] = mapped_column(Numeric, nullable=True)
    gate_passed: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    metrics: Mapped[Optional[dict[str, Any]]] = mapped_column(JSONB, nullable=True)
    per_metric_breakdown: Mapped[Optional[dict[str, Any]]] = mapped_column(JSONB, nullable=True)
    key_error_moments: Mapped[Optional[list[dict[str, Any]]]] = mapped_column(JSONB, nullable=True)
    semantic_phases: Mapped[Optional[dict[str, Any]]] = mapped_column(JSONB, nullable=True)
    gate_reasons: Mapped[Optional[list[str]]] = mapped_column(JSONB, nullable=True)
    summary_text: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=False), server_default=func.now(), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=False),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )

    # Backward-compatible aliases used by existing service code.
    score = synonym("overall_score")

    attempt = relationship("LearnerAttempt", back_populates="evaluation_result")
    expert_video = relationship(
        "ExpertVideo",
        foreign_keys=[expert_video_id],
        back_populates="expert_evaluations",
    )
    learner_video = relationship(
        "ExpertVideo",
        foreign_keys=[learner_video_id],
        back_populates="learner_evaluations",
    )
    feedback = relationship("EvaluationFeedback", back_populates="evaluation", uselist=False)


# Canonical model name expected by the current PostgreSQL schema.
Evaluation = EvaluationResult

# Ensure relationship target is registered when this module is imported.
import app.models.evaluation_feedback  # noqa: E402,F401
