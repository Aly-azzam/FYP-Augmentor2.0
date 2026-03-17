from fastapi import APIRouter, Depends, Query
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db
from app.schemas.evaluation_schema import HistoryEntry
from app.services.history_service import get_history

router = APIRouter(prefix="/api/history", tags=["History"])


@router.get("", response_model=list[HistoryEntry])
async def list_history(
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    db: AsyncSession = Depends(get_db),
):
    """Get evaluation history derived from LearnerAttempt + EvaluationResult."""
    return await get_history(db, limit=limit, offset=offset)
