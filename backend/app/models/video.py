import uuid
from datetime import datetime
from decimal import Decimal

from sqlalchemy import BigInteger, DateTime, ForeignKey, Integer, Numeric, String, Text, func
from sqlalchemy.dialects.postgresql import UUID as PG_UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base


class Video(Base):
    __tablename__ = "videos"

    id: Mapped[str] = mapped_column(
        PG_UUID(as_uuid=False),
        primary_key=True,
        default=lambda: str(uuid.uuid4()),
    )
    owner_user_id: Mapped[str | None] = mapped_column(
        PG_UUID(as_uuid=False),
        ForeignKey("users.id", ondelete="SET NULL"),
        nullable=True,
    )
    chapter_id: Mapped[str | None] = mapped_column(
        PG_UUID(as_uuid=False),
        ForeignKey("chapters.id", ondelete="SET NULL"),
        nullable=True,
    )
    video_role: Mapped[str] = mapped_column(String(50), nullable=False)
    source_video_id: Mapped[str | None] = mapped_column(
        PG_UUID(as_uuid=False),
        ForeignKey("videos.id", ondelete="SET NULL"),
        nullable=True,
    )
    file_path: Mapped[str] = mapped_column(Text, nullable=False)
    file_name: Mapped[str | None] = mapped_column(String(255), nullable=True)
    mime_type: Mapped[str | None] = mapped_column(String(100), nullable=True)
    duration_seconds: Mapped[Decimal | None] = mapped_column(Numeric, nullable=True)
    fps: Mapped[Decimal | None] = mapped_column(Numeric, nullable=True)
    width: Mapped[int | None] = mapped_column(Integer, nullable=True)
    height: Mapped[int | None] = mapped_column(Integer, nullable=True)
    file_size_bytes: Mapped[int | None] = mapped_column(BigInteger, nullable=True)
    storage_provider: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
        server_default="local",
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )

    owner_user: Mapped["User | None"] = relationship("User", back_populates="videos")
    chapter: Mapped["Chapter | None"] = relationship("Chapter", back_populates="videos")
    source_video: Mapped["Video | None"] = relationship(
        "Video",
        remote_side=[id],
        back_populates="derived_videos",
    )
    derived_videos: Mapped[list["Video"]] = relationship(
        "Video",
        back_populates="source_video",
    )

    attempts: Mapped[list["Attempt"]] = relationship(
        "Attempt",
        back_populates="learner_video",
    )
    expert_evaluations: Mapped[list["Evaluation"]] = relationship(
        "Evaluation",
        foreign_keys="Evaluation.expert_video_id",
        back_populates="expert_video",
    )
    learner_evaluations: Mapped[list["Evaluation"]] = relationship(
        "Evaluation",
        foreign_keys="Evaluation.learner_video_id",
        back_populates="learner_video",
    )

