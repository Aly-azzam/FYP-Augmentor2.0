import uuid
from datetime import datetime
from typing import Any, Optional

from sqlalchemy import DateTime, ForeignKey, String, Text, func
from sqlalchemy.dialects.postgresql import JSONB, UUID as PG_UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base


class EvaluationFeedback(Base):
    __tablename__ = "evaluation_feedback"

    id: Mapped[str] = mapped_column(
        PG_UUID(as_uuid=False),
        primary_key=True,
        default=lambda: str(uuid.uuid4()),
    )
    evaluation_id: Mapped[str] = mapped_column(
        PG_UUID(as_uuid=False),
        ForeignKey("evaluations.id", ondelete="CASCADE"),
        nullable=False,
        unique=True,
    )
    mode: Mapped[str] = mapped_column(String(50), nullable=False)
    model_name: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    explanation_text: Mapped[str] = mapped_column(Text, nullable=False)
    strengths: Mapped[Optional[list[str]]] = mapped_column(JSONB, nullable=True)
    weaknesses: Mapped[Optional[list[str]]] = mapped_column(JSONB, nullable=True)
    advice: Mapped[Optional[dict[str, Any] | list[str] | str]] = mapped_column(JSONB, nullable=True)
    cited_timestamps: Mapped[Optional[list[dict[str, Any]]]] = mapped_column(JSONB, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    evaluation: Mapped["Evaluation"] = relationship("Evaluation", back_populates="feedback")
