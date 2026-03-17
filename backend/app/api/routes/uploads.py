import uuid

from fastapi import APIRouter, UploadFile, File, Form

from app.schemas.upload_schema import UploadResponse

router = APIRouter(prefix="/api/uploads", tags=["Uploads"])


@router.post("/practice-video", response_model=UploadResponse)
async def upload_practice_video(
    chapter_id: str = Form(...),
    file: UploadFile = File(...),
):
    """Upload a learner practice video for a specific chapter.

    Creates a LearnerAttempt record and stores the video file.
    """
    attempt_id = uuid.uuid4()

    # TODO: implement real flow:
    # 1. Create LearnerAttempt with status=created
    # 2. Validate video (format, duration)
    # 3. Save file to storage
    # 4. Update status to uploaded (or failed if validation fails)

    return UploadResponse(
        attempt_id=attempt_id,
        status="uploaded",
        message="Video uploaded successfully. Ready for evaluation.",
    )
