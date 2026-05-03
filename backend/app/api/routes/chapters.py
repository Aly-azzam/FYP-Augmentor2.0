from uuid import UUID

from pathlib import Path
from uuid import uuid4

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status
from sqlalchemy import func, select
from sqlalchemy.orm import Session, selectinload

from app.core.config import settings
from app.core.database import get_db
from app.models.chapter import Chapter
from app.models.course import Course
from app.models.video import Video
from app.schemas.course_schema import (
    ChapterCreateRequest,
    ChapterDetail,
    ExpertVideoAssetOut,
    ExpertVideoOut,
    ExpertVideoUploadResponse,
)
from app.services.media_service import (
    build_storage_url,
    get_default_expert_video_asset,
    normalize_storage_key,
    storage_path_exists,
)
from app.services.upload_service import UploadValidationError, save_upload_file, validate_upload_file

router = APIRouter(prefix="/api/chapters", tags=["Chapters"])

DEFAULT_EXPERT_UPLOAD_COURSE_TITLE = "Cut a straight line"


def _chapter_detail(chapter: Chapter) -> ChapterDetail:
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
        order=chapter.chapter_order,
        expert_video=expert_video,
    )


@router.get("", response_model=list[ChapterDetail])
def list_chapters(course_id: UUID | None = None, db: Session = Depends(get_db)):
    """List chapters, optionally filtered by course_id."""
    query = select(Chapter).options(selectinload(Chapter.expert_video)).order_by(Chapter.chapter_order.asc())
    if course_id is not None:
        query = query.where(Chapter.course_id == course_id)

    chapters = db.execute(query).scalars().all()
    return [_chapter_detail(chapter) for chapter in chapters]


@router.post("", response_model=ChapterDetail, status_code=status.HTTP_201_CREATED)
def create_chapter(payload: ChapterCreateRequest, db: Session = Depends(get_db)):
    """Create a chapter so an expert video can be uploaded for it."""
    title = payload.title.strip()
    if not title:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Chapter title is required.",
        )

    if payload.course_id is not None:
        course = db.execute(
            select(Course).where(Course.id == str(payload.course_id))
        ).scalar_one_or_none()
        if course is None:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Course not found.")
    else:
        course_title = (payload.course_title or DEFAULT_EXPERT_UPLOAD_COURSE_TITLE).strip()
        course = db.execute(
            select(Course).where(Course.title == course_title)
        ).scalar_one_or_none()
        if course is None:
            course = Course(
                title=course_title,
                description="Expert upload course for cutting practice.",
            )
            db.add(course)
            db.flush()

    chapter_order = payload.order
    if chapter_order is None:
        max_order = db.execute(
            select(func.max(Chapter.chapter_order)).where(Chapter.course_id == course.id)
        ).scalar_one_or_none()
        chapter_order = int(max_order or 0) + 1

    chapter = Chapter(
        course_id=course.id,
        title=title,
        description=payload.description,
        chapter_order=chapter_order,
    )
    db.add(chapter)
    db.commit()
    db.refresh(chapter)
    return _chapter_detail(chapter)


@router.get("/default/expert-video", response_model=ExpertVideoAssetOut)
def get_default_expert_video(db: Session = Depends(get_db)):
    """Return the default expert video for frontend playback."""
    asset = get_default_expert_video_asset(db)
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
def get_chapter(chapter_id: UUID, db: Session = Depends(get_db)):
    """Get chapter details."""
    result = db.execute(
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
        order=chapter.chapter_order,
        expert_video=expert_video,
    )


@router.get("/{chapter_id}/expert-video", response_model=ExpertVideoOut)
def get_expert_video(chapter_id: UUID, db: Session = Depends(get_db)):
    """Get expert reference video for a chapter."""
    result = db.execute(
        select(Video).where(
            Video.chapter_id == chapter_id,
            Video.video_role == "expert",
        )
    )
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


@router.post("/{chapter_id}/expert-video", response_model=ExpertVideoUploadResponse)
async def upload_expert_video(
    chapter_id: UUID,
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
):
    """Upload or replace the expert reference video for a chapter."""
    chapter = db.execute(select(Chapter).where(Chapter.id == chapter_id)).scalar_one_or_none()
    if chapter is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Chapter not found.")

    try:
        extension = validate_upload_file(file)
    except UploadValidationError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc

    relative_path = Path("expert") / str(chapter_id) / f"{uuid4()}{extension}"
    old_storage_key: str | None = None

    existing_expert = db.execute(
        select(Video).where(
            Video.chapter_id == str(chapter_id),
            Video.video_role == "expert",
        )
    ).scalar_one_or_none()
    if existing_expert is not None:
        try:
            old_storage_key = normalize_storage_key(existing_expert.file_path)
        except ValueError:
            old_storage_key = None

    try:
        file_size_bytes = await save_upload_file(file, relative_path)
    except UploadValidationError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to store uploaded expert video.",
        ) from exc

    storage_key = relative_path.as_posix()
    try:
        if existing_expert is None:
            expert_video = Video(
                owner_user_id=None,
                chapter_id=str(chapter_id),
                video_role="expert",
                source_video_id=None,
                file_path=storage_key,
                file_name=file.filename,
                mime_type=(file.content_type or None),
                file_size_bytes=file_size_bytes,
                storage_provider="local",
            )
            db.add(expert_video)
            db.flush()
        else:
            existing_expert.file_path = storage_key
            existing_expert.file_name = file.filename
            existing_expert.mime_type = file.content_type or None
            existing_expert.file_size_bytes = file_size_bytes
            expert_video = existing_expert

        db.commit()
        db.refresh(expert_video)
    except Exception as exc:
        db.rollback()
        saved_file = settings.STORAGE_ROOT / relative_path
        if saved_file.exists():
            saved_file.unlink()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to persist uploaded expert video.",
        ) from exc

    if old_storage_key and old_storage_key != storage_key:
        old_path = settings.STORAGE_ROOT / old_storage_key
        if old_path.exists():
            old_path.unlink()

    return ExpertVideoUploadResponse(
        message="Expert video uploaded successfully.",
        chapter_id=chapter_id,
        expert_video_id=UUID(str(expert_video.id)),
        file_path=storage_key,
        url=build_storage_url(storage_key),
    )
