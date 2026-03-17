import uuid
from datetime import datetime
from typing import Optional

from sqlalchemy import String, Float, Integer, ForeignKey, DateTime, Text, func
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.core.database import Base


class EvaluationResult(Base):
    __tablename__ = "evaluation_results"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    attempt_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("learner_attempts.id"), unique=True, nullable=False
    )

    score: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    angle_deviation: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    trajectory_deviation: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    velocity_difference: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    tool_alignment_deviation: Mapped[Optional[float]] = mapped_column(Float, nullable=True)

    summary: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    ai_text: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    warnings: Mapped[Optional[dict]] = mapped_column(JSONB, nullable=True)
    comparison_media: Mapped[Optional[dict]] = mapped_column(JSONB, nullable=True)

    # Artifact pointer
    result_path: Mapped[Optional[str]] = mapped_column(String(512), nullable=True)

    # Version metadata for reproducibility
    pipeline_version: Mapped[Optional[str]] = mapped_column(String(32), nullable=True)
    model_version: Mapped[Optional[str]] = mapped_column(String(32), nullable=True)
    config_version: Mapped[Optional[str]] = mapped_column(String(32), nullable=True)

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    attempt = relationship("LearnerAttempt", back_populates="evaluation_result")
