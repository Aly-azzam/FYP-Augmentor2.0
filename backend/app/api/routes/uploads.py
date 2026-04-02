from typing import Optional
from uuid import UUID

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.schemas.upload_schema import UploadResponse
from app.services.media_service import build_storage_url
from app.services.upload_service import (
    RelatedResourceNotFoundError,
    UploadPersistenceError,
    UploadValidationError,
    create_learner_attempt_upload,
)

router = APIRouter(prefix="/api/uploads", tags=["Uploads"])


@router.post("/practice-video", response_model=UploadResponse)
async def upload_practice_video(
    chapter_id: UUID = Form(...),
    file: UploadFile = File(...),
    user_id: Optional[UUID] = Form(None),
    db: Session = Depends(get_db),
):
    """Upload a learner practice video and persist its attempt record."""
    try:
        result = await create_learner_attempt_upload(
            db,
            chapter_id=chapter_id,
            file=file,
            user_id=user_id,
        )
    except RelatedResourceNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    except UploadValidationError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    except UploadPersistenceError as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to store uploaded video.",
        ) from exc

    return UploadResponse(
        attempt_id=result.attempt.id,
        chapter_id=result.attempt.chapter_id,
        expert_video_id=result.expert_video_id,
        upload_status=result.attempt.status,
        original_filename=result.attempt.original_filename or file.filename or "",
        stored_path=result.attempt.video_path or "",
        video_url=build_storage_url(result.attempt.video_path) if result.attempt.video_path else None,
        message="Video uploaded successfully. Learner attempt is ready for evaluation.",
    )
