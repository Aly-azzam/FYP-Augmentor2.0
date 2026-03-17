from fastapi import APIRouter

from app.schemas.evaluation_schema import ProgressOut

router = APIRouter(prefix="/api/progress", tags=["Progress"])


@router.get("", response_model=list[ProgressOut])
async def list_progress():
    """Get completion-based progress per course."""
    # TODO: query real DB via progress_service
    return []
