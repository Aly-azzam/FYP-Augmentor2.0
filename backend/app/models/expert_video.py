import uuid
from datetime import datetime

from sqlalchemy import BigInteger, DateTime, ForeignKey, Integer, Numeric, String, Text, func
from sqlalchemy.dialects.postgresql import UUID as PG_UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.core.database import Base


class ExpertVideo(Base):
    __tablename__ = "videos"

    id: Mapped[str] = mapped_column(
        PG_UUID(as_uuid=False),
        primary_key=True,
        default=lambda: str(uuid.uuid4()),
    )
    owner_user_id: Mapped[str | None] = mapped_column(
        PG_UUID(as_uuid=False), ForeignKey("users.id"), nullable=True
    )
    chapter_id: Mapped[str | None] = mapped_column(
        PG_UUID(as_uuid=False), ForeignKey("chapters.id"), nullable=True
    )
    video_role: Mapped[str] = mapped_column(String(50), nullable=False)
    source_video_id: Mapped[str | None] = mapped_column(
        PG_UUID(as_uuid=False), ForeignKey("videos.id"), nullable=True
    )
    file_path: Mapped[str] = mapped_column(Text, nullable=False)
    file_name: Mapped[str | None] = mapped_column(String(255), nullable=True)
    mime_type: Mapped[str | None] = mapped_column(String(100), nullable=True)
    duration_seconds: Mapped[float | None] = mapped_column(Numeric, nullable=True)
    fps: Mapped[float | None] = mapped_column(Numeric, nullable=True)
    width: Mapped[int | None] = mapped_column(Integer, nullable=True)
    height: Mapped[int | None] = mapped_column(Integer, nullable=True)
    file_size_bytes: Mapped[int | None] = mapped_column(BigInteger, nullable=True)
    storage_provider: Mapped[str] = mapped_column(String(50), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=False), server_default=func.now(), nullable=False
    )

    chapter = relationship("Chapter", back_populates="videos")
    owner_user = relationship("User", back_populates="owned_videos")
    source_video = relationship(
        "ExpertVideo",
        remote_side="ExpertVideo.id",
        back_populates="derived_videos",
    )
    derived_videos = relationship("ExpertVideo", back_populates="source_video")

    attempts = relationship("LearnerAttempt", back_populates="learner_video")
    expert_evaluations = relationship(
        "EvaluationResult",
        foreign_keys="EvaluationResult.expert_video_id",
        back_populates="expert_video",
    )
    learner_evaluations = relationship(
        "EvaluationResult",
        foreign_keys="EvaluationResult.learner_video_id",
        back_populates="learner_video",
    )


# Canonical model name expected by the current PostgreSQL schema.
Video = ExpertVideo
