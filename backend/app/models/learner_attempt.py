import uuid
from datetime import datetime
from typing import Optional

from sqlalchemy import String, ForeignKey, DateTime, Text, func
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.core.database import Base


class LearnerAttempt(Base):
    """Central entity and source of truth for the processing lifecycle."""

    __tablename__ = "learner_attempts"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    chapter_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("chapters.id"), nullable=False)
    user_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        UUID(as_uuid=True), ForeignKey("users.id"), nullable=True
    )

    status: Mapped[str] = mapped_column(String(32), nullable=False, default="created")
    original_filename: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    content_type: Mapped[Optional[str]] = mapped_column(String(128), nullable=True)
    file_size_bytes: Mapped[Optional[int]] = mapped_column(nullable=True)
    video_path: Mapped[Optional[str]] = mapped_column(String(512), nullable=True)
    error_reason: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    # Artifact pointers
    landmarks_path: Mapped[Optional[str]] = mapped_column(String(512), nullable=True)
    motion_path: Mapped[Optional[str]] = mapped_column(String(512), nullable=True)

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )

    chapter = relationship("Chapter", back_populates="attempts")
    user = relationship("User", back_populates="attempts")
    evaluation_result = relationship("EvaluationResult", back_populates="attempt", uselist=False)
