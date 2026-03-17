"""Progress Service — derive completion-based progress.

Progress rule:
- chapter is complete if at least one attempt is completed or completed_with_warnings
- course progress = completed chapters / total chapters
- failed attempts do not count
- progress never decreases
"""

from uuid import UUID
from typing import Optional

from sqlalchemy.ext.asyncio import AsyncSession

from app.schemas.evaluation_schema import ProgressOut


async def get_progress(
    db: AsyncSession,
    user_id: Optional[UUID] = None,
) -> list[ProgressOut]:
    """Compute completion-based progress per course.

    Returns list of ProgressOut.
    """
    # TODO: implement real query deriving completion from attempts
    return []
