import uuid
from datetime import datetime
from typing import Optional

from sqlalchemy import DateTime, ForeignKey, String, Text, func
from sqlalchemy.dialects.postgresql import UUID as PG_UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.core.database import Base


class LearnerAttempt(Base):
    """Central entity and source of truth for the processing lifecycle."""

    __tablename__ = "attempts"

    id: Mapped[str] = mapped_column(
        PG_UUID(as_uuid=False),
        primary_key=True,
        default=lambda: str(uuid.uuid4()),
    )
    user_id: Mapped[str] = mapped_column(
        PG_UUID(as_uuid=False), ForeignKey("users.id"), nullable=False
    )
    chapter_id: Mapped[str] = mapped_column(
        PG_UUID(as_uuid=False), ForeignKey("chapters.id"), nullable=False
    )
    learner_video_id: Mapped[str] = mapped_column(
        PG_UUID(as_uuid=False), ForeignKey("videos.id"), nullable=False
    )

    status: Mapped[str] = mapped_column(String(50), nullable=False)
    original_filename: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    error_reason: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=False), server_default=func.now(), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=False),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )

    chapter = relationship("Chapter", back_populates="attempts")
    user = relationship("User", back_populates="attempts")
    learner_video = relationship("ExpertVideo", back_populates="attempts")
    evaluation_result = relationship("EvaluationResult", back_populates="attempt", uselist=False)


# Canonical model name expected by the current PostgreSQL schema.
Attempt = LearnerAttempt
