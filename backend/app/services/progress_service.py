from __future__ import annotations


def compute_progress(evaluations: list[dict]) -> dict:
    """Compute simple score progress statistics for one attempt."""
    if not evaluations:
        return {
            "best_score": 0.0,
            "average_score": 0.0,
            "attempt_count": 0,
            "latest_score": 0.0,
        }

    scores = [float(item.get("score", 0.0)) for item in evaluations]
    return {
        "best_score": round(max(scores), 4),
        "average_score": round(sum(scores) / float(len(scores)), 4),
        "attempt_count": len(scores),
        "latest_score": round(float(scores[0]), 4),
    }

"""Progress Service — derive completion-based progress.

Progress rule:
- chapter is complete if at least one attempt is completed or completed_with_warnings
- course progress = completed chapters / total chapters
- failed attempts do not count
- progress never decreases
"""

from uuid import UUID
from typing import Optional

from sqlalchemy.orm import Session

from app.schemas.evaluation_schema import ProgressOut


async def get_progress(
    db: Session,
    user_id: Optional[UUID] = None,
) -> list[ProgressOut]:
    """Compute completion-based progress per course.

    Returns list of ProgressOut.
    """
    # TODO: implement real query deriving completion from attempts
    return []
