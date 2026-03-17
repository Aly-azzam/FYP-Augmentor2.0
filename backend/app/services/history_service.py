"""History Service — derive history from LearnerAttempt + EvaluationResult.

History is not stored separately. It is queried/derived from existing tables.
"""

from uuid import UUID
from typing import Optional

from sqlalchemy.ext.asyncio import AsyncSession

from app.schemas.evaluation_schema import HistoryEntry


async def get_history(
    db: AsyncSession,
    user_id: Optional[UUID] = None,
    limit: int = 50,
    offset: int = 0,
) -> list[HistoryEntry]:
    """Derive evaluation history from attempts and results.

    Returns list of HistoryEntry sorted by most recent first.
    """
    # TODO: implement real query joining learner_attempts, evaluation_results, chapters, courses
    return []
