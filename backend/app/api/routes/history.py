from fastapi import APIRouter, Query

from app.schemas.evaluation_schema import HistoryEntry

router = APIRouter(prefix="/api/history", tags=["History"])


@router.get("", response_model=list[HistoryEntry])
async def list_history(
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
):
    """Get evaluation history derived from LearnerAttempt + EvaluationResult."""
    # TODO: query real DB via history_service
    return []
