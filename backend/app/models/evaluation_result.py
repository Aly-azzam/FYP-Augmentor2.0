import uuid
from datetime import datetime
from typing import Any, Optional

from sqlalchemy import String, Float, ForeignKey, DateTime, JSON, func
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.core.database import Base


class EvaluationResult(Base):
    __tablename__ = "evaluation_results"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    attempt_id: Mapped[str] = mapped_column(String(36), ForeignKey("learner_attempts.id"), nullable=False, index=True)
    score: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    metrics: Mapped[Optional[dict[str, Any]]] = mapped_column(JSON, nullable=True)
    per_metric_breakdown: Mapped[Optional[dict[str, Any]]] = mapped_column(JSON, nullable=True)
    key_error_moments: Mapped[Optional[list[dict[str, Any]]]] = mapped_column(JSON, nullable=True)
    semantic_phases: Mapped[Optional[dict[str, Any]]] = mapped_column(JSON, nullable=True)
    explanation: Mapped[Optional[dict[str, Any]]] = mapped_column(JSON, nullable=True)

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    attempt = relationship("LearnerAttempt", back_populates="evaluation_result")
