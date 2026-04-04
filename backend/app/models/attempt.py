import uuid
from datetime import datetime

from sqlalchemy import DateTime, ForeignKey, String, Text, func
from sqlalchemy.dialects.postgresql import UUID as PG_UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base


class Attempt(Base):
    __tablename__ = "attempts"

    id: Mapped[str] = mapped_column(
        PG_UUID(as_uuid=False),
        primary_key=True,
        default=lambda: str(uuid.uuid4()),
    )
    user_id: Mapped[str] = mapped_column(
        PG_UUID(as_uuid=False),
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
    )
    chapter_id: Mapped[str] = mapped_column(
        PG_UUID(as_uuid=False),
        ForeignKey("chapters.id", ondelete="CASCADE"),
        nullable=False,
    )
    learner_video_id: Mapped[str] = mapped_column(
        PG_UUID(as_uuid=False),
        ForeignKey("videos.id", ondelete="CASCADE"),
        nullable=False,
    )
    status: Mapped[str] = mapped_column(String(50), nullable=False)
    original_filename: Mapped[str | None] = mapped_column(String(255), nullable=True)
    error_reason: Mapped[str | None] = mapped_column(Text, nullable=True)
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

    user: Mapped["User"] = relationship("User", back_populates="attempts")
    chapter: Mapped["Chapter"] = relationship("Chapter", back_populates="attempts")
    learner_video: Mapped["Video"] = relationship("Video", back_populates="attempts")
    evaluation: Mapped["Evaluation | None"] = relationship(
        "Evaluation",
        back_populates="attempt",
        uselist=False,
    )

