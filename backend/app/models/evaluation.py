import uuid
from datetime import datetime
from decimal import Decimal
from typing import Any

from sqlalchemy import Boolean, DateTime, ForeignKey, Numeric, String, Text, func
from sqlalchemy.dialects.postgresql import JSONB, UUID as PG_UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship, synonym

from app.db.base import Base


class Evaluation(Base):
    __tablename__ = "evaluations"

    id: Mapped[str] = mapped_column(
        PG_UUID(as_uuid=False),
        primary_key=True,
        default=lambda: str(uuid.uuid4()),
    )
    attempt_id: Mapped[str] = mapped_column(
        PG_UUID(as_uuid=False),
        ForeignKey("attempts.id", ondelete="CASCADE"),
        nullable=False,
        unique=True,
    )
    expert_video_id: Mapped[str] = mapped_column(
        PG_UUID(as_uuid=False),
        ForeignKey("videos.id", ondelete="CASCADE"),
        nullable=False,
    )
    learner_video_id: Mapped[str] = mapped_column(
        PG_UUID(as_uuid=False),
        ForeignKey("videos.id", ondelete="CASCADE"),
        nullable=False,
    )
    overall_score: Mapped[Decimal | None] = mapped_column(Numeric, nullable=True)
    status: Mapped[str] = mapped_column(String(50), nullable=False)
    context_confidence: Mapped[Decimal | None] = mapped_column(Numeric, nullable=True)
    score_confidence: Mapped[Decimal | None] = mapped_column(Numeric, nullable=True)
    explanation_confidence: Mapped[Decimal | None] = mapped_column(Numeric, nullable=True)
    gate_passed: Mapped[bool] = mapped_column(Boolean, nullable=False, server_default="true")
    gate_reasons: Mapped[list[Any] | dict[str, Any] | None] = mapped_column(JSONB, nullable=True)
    metrics: Mapped[dict[str, Any] | None] = mapped_column(JSONB, nullable=True)
    per_metric_breakdown: Mapped[dict[str, Any] | None] = mapped_column(JSONB, nullable=True)
    key_error_moments: Mapped[list[dict[str, Any]] | None] = mapped_column(JSONB, nullable=True)
    semantic_phases: Mapped[dict[str, Any] | None] = mapped_column(JSONB, nullable=True)
    summary_text: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )

    # Backward-compatible alias used by legacy service code.
    score = synonym("overall_score")

    attempt: Mapped["Attempt"] = relationship("Attempt", back_populates="evaluation", uselist=False)
    expert_video: Mapped["Video"] = relationship(
        "Video",
        foreign_keys=[expert_video_id],
        back_populates="expert_evaluations",
    )
    learner_video: Mapped["Video"] = relationship(
        "Video",
        foreign_keys=[learner_video_id],
        back_populates="learner_evaluations",
    )
    feedback: Mapped["EvaluationFeedback | None"] = relationship(
        "EvaluationFeedback",
        back_populates="evaluation",
        uselist=False,
    )

