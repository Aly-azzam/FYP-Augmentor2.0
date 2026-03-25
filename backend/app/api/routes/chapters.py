from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.core.database import get_db
from app.models.chapter import Chapter
from app.models.expert_video import ExpertVideo
from app.schemas.course_schema import ChapterDetail, ExpertVideoAssetOut, ExpertVideoOut
from app.services.media_service import (
    build_storage_url,
    get_default_expert_video_asset,
    normalize_storage_key,
    storage_path_exists,
)

router = APIRouter(prefix="/api/chapters", tags=["Chapters"])


@router.get("/default/expert-video", response_model=ExpertVideoAssetOut)
async def get_default_expert_video(db: AsyncSession = Depends(get_db)):
    """Return the default expert video for frontend playback."""
    asset = await get_default_expert_video_asset(db)
    if asset is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No expert video found in the database or storage.",
        )

    return ExpertVideoAssetOut(
        filename=asset.filename,
        storage_key=asset.storage_key,
        file_path=asset.storage_key,
        url=asset.url,
        chapter_id=asset.chapter_id,
        expert_video_id=asset.expert_video_id,
    )


@router.get("/{chapter_id}", response_model=ChapterDetail)
async def get_chapter(chapter_id: UUID, db: AsyncSession = Depends(get_db)):
    """Get chapter details."""
    result = await db.execute(
        select(Chapter)
        .options(selectinload(Chapter.expert_video))
        .where(Chapter.id == chapter_id)
    )
    chapter = result.scalar_one_or_none()
    if chapter is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Chapter not found.")

    expert_video = None
    if chapter.expert_video is not None:
        try:
            storage_key = normalize_storage_key(chapter.expert_video.file_path)
        except ValueError:
            storage_key = None

        if storage_key is not None:
            expert_video = ExpertVideoOut(
                id=chapter.expert_video.id,
                chapter_id=chapter.expert_video.chapter_id,
                file_path=storage_key,
                url=build_storage_url(storage_key),
                duration_seconds=chapter.expert_video.duration_seconds,
                fps=chapter.expert_video.fps,
            )

    return ChapterDetail(
        id=chapter.id,
        course_id=chapter.course_id,
        title=chapter.title,
        order=chapter.order,
        expert_video=expert_video,
    )


@router.get("/{chapter_id}/expert-video", response_model=ExpertVideoOut)
async def get_expert_video(chapter_id: UUID, db: AsyncSession = Depends(get_db)):
    """Get expert reference video for a chapter."""
    result = await db.execute(select(ExpertVideo).where(ExpertVideo.chapter_id == chapter_id))
    expert_video = result.scalar_one_or_none()
    if expert_video is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Expert video not found for the requested chapter.",
        )

    try:
        storage_key = normalize_storage_key(expert_video.file_path)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Expert video path is invalid.",
        ) from exc

    if not storage_path_exists(storage_key):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Expert video file is missing from storage.",
        )

    return ExpertVideoOut(
        id=expert_video.id,
        chapter_id=expert_video.chapter_id,
        file_path=storage_key,
        url=build_storage_url(storage_key),
        duration_seconds=expert_video.duration_seconds,
        fps=expert_video.fps,
    )
