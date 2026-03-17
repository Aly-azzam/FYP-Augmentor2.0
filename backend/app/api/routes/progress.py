from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db
from app.schemas.evaluation_schema import ProgressOut
from app.services.progress_service import get_progress

router = APIRouter(prefix="/api/progress", tags=["Progress"])


@router.get("", response_model=list[ProgressOut])
async def list_progress(
    db: AsyncSession = Depends(get_db),
):
    """Get completion-based progress per course."""
    return await get_progress(db)
