import uuid
from datetime import datetime

from sqlalchemy import DateTime, ForeignKey, Integer, String, Text, and_, func
from sqlalchemy.dialects.postgresql import UUID as PG_UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship, synonym

from app.db.base import Base


class Chapter(Base):
    __tablename__ = "chapters"

    id: Mapped[str] = mapped_column(
        PG_UUID(as_uuid=False),
        primary_key=True,
        default=lambda: str(uuid.uuid4()),
    )
    course_id: Mapped[str] = mapped_column(
        PG_UUID(as_uuid=False), ForeignKey("courses.id"), nullable=False
    )
    title: Mapped[str] = mapped_column(String(255), nullable=False)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    chapter_order: Mapped[int] = mapped_column(Integer, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )

    # Backward-compatible alias used by legacy query code.
    order = synonym("chapter_order")

    course: Mapped["Course"] = relationship("Course", back_populates="chapters")
    videos: Mapped[list["Video"]] = relationship("Video", back_populates="chapter")
    expert_video: Mapped["Video | None"] = relationship(
        "Video",
        primaryjoin="and_(Chapter.id==Video.chapter_id, Video.video_role=='expert')",
        uselist=False,
        viewonly=True,
    )
    attempts: Mapped[list["Attempt"]] = relationship("Attempt", back_populates="chapter")
