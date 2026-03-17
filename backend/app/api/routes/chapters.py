from uuid import UUID

from fastapi import APIRouter

from app.schemas.course_schema import ChapterDetail, ExpertVideoOut

router = APIRouter(prefix="/api/chapters", tags=["Chapters"])


@router.get("/{chapter_id}", response_model=ChapterDetail)
async def get_chapter(chapter_id: UUID):
    """Get chapter details."""
    # TODO: query real DB, return 404 if not found
    return ChapterDetail(
        id=chapter_id,
        course_id="00000000-0000-0000-0000-000000000000",
        title="Placeholder Chapter",
        order=1,
    )


@router.get("/{chapter_id}/expert-video", response_model=ExpertVideoOut)
async def get_expert_video(chapter_id: UUID):
    """Get expert reference video for a chapter."""
    # TODO: query real DB, return 404 if not found
    return ExpertVideoOut(
        id="00000000-0000-0000-0000-000000000001",
        chapter_id=chapter_id,
        file_path="/storage/expert/placeholder.mp4",
    )
