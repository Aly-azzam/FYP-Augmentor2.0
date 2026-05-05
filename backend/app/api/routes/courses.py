from uuid import UUID

from pathlib import Path
from uuid import uuid4

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status
from sqlalchemy import select
from sqlalchemy.orm import Session, selectinload

from app.core.config import settings
from app.core.database import get_db
from app.models.chapter import Chapter
from app.models.course import Course
from app.schemas.course_schema import CourseCreateRequest, CourseOut, CourseDetail
from app.services.media_service import build_storage_url

router = APIRouter(prefix="/api/courses", tags=["Courses"])

COURSE_THUMBNAIL_ROOT = Path("course_thumbnails")
ALLOWED_THUMBNAIL_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".gif"}
ALLOWED_THUMBNAIL_TYPES = {
    "image/jpeg",
    "image/png",
    "image/webp",
    "image/gif",
}
MAX_THUMBNAIL_BYTES = 5 * 1024 * 1024
UPLOAD_CHUNK_BYTES = 1024 * 1024


def _course_thumbnail_url(course_id: str | UUID) -> str | None:
    thumbnail_dir = settings.STORAGE_ROOT / COURSE_THUMBNAIL_ROOT / str(course_id)
    if not thumbnail_dir.exists():
        return None

    for thumbnail_path in sorted(thumbnail_dir.iterdir()):
        if thumbnail_path.is_file() and thumbnail_path.suffix.lower() in ALLOWED_THUMBNAIL_EXTENSIONS:
            return build_storage_url(thumbnail_path)
    return None


def _course_out(course: Course) -> CourseOut:
    return CourseOut(
        id=course.id,
        title=course.title,
        description=course.description,
        created_at=course.created_at,
        updated_at=course.updated_at,
        thumbnail_url=_course_thumbnail_url(course.id),
    )


@router.get("", response_model=list[CourseOut])
def list_courses(db: Session = Depends(get_db)):
    """List all available courses."""
    result = db.execute(select(Course).order_by(Course.created_at.desc()))
    courses = result.scalars().all()
    return [_course_out(course) for course in courses]


@router.post("", response_model=CourseOut, status_code=status.HTTP_201_CREATED)
def create_course(payload: CourseCreateRequest, db: Session = Depends(get_db)):
    """Create a course."""
    title = payload.title.strip()
    if not title:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Course title is required.",
        )

    existing = db.execute(select(Course).where(Course.title == title)).scalar_one_or_none()
    if existing is not None:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="A course with this title already exists.",
        )

    course = Course(
        title=title,
        description=payload.description.strip() if payload.description else None,
    )
    db.add(course)
    db.commit()
    db.refresh(course)
    return _course_out(course)


@router.post("/{course_id}/thumbnail", response_model=CourseOut)
async def upload_course_thumbnail(
    course_id: UUID,
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
):
    """Upload or replace the display image used by a course card."""
    course = db.execute(select(Course).where(Course.id == course_id)).scalar_one_or_none()
    if course is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Course not found.")

    filename = (file.filename or "").strip()
    extension = Path(filename).suffix.lower()
    if extension not in ALLOWED_THUMBNAIL_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Unsupported image format. Use jpg, png, webp, or gif.",
        )

    content_type = (file.content_type or "").strip().lower()
    if content_type and content_type not in ALLOWED_THUMBNAIL_TYPES:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Unsupported image content type.",
        )

    thumbnail_dir = settings.STORAGE_ROOT / COURSE_THUMBNAIL_ROOT / str(course_id)
    thumbnail_dir.mkdir(parents=True, exist_ok=True)
    destination = thumbnail_dir / f"{uuid4()}{extension}"

    bytes_written = 0
    try:
        with destination.open("wb") as output:
            while chunk := await file.read(UPLOAD_CHUNK_BYTES):
                bytes_written += len(chunk)
                if bytes_written > MAX_THUMBNAIL_BYTES:
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail="Course image must be 5 MB or smaller.",
                    )
                output.write(chunk)
    except Exception:
        if destination.exists():
            destination.unlink()
        raise
    finally:
        await file.close()

    if bytes_written == 0:
        if destination.exists():
            destination.unlink()
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Uploaded image is empty.")

    for thumbnail_path in thumbnail_dir.iterdir():
        if thumbnail_path != destination and thumbnail_path.is_file():
            try:
                thumbnail_path.unlink()
            except OSError:
                pass

    return _course_out(course)


@router.get("/{course_id}", response_model=CourseDetail)
def get_course(course_id: UUID, db: Session = Depends(get_db)):
    """Get course details including chapters."""
    result = db.execute(
        select(Course)
        .options(selectinload(Course.chapters).selectinload(Chapter.expert_video))
        .where(Course.id == course_id)
    )
    course = result.scalar_one_or_none()
    if course is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Course not found.")

    chapter_briefs = [
        {
            "id": chapter.id,
            "title": chapter.title,
            "order": chapter.chapter_order,
            "has_expert_video": chapter.expert_video is not None,
        }
        for chapter in sorted(course.chapters, key=lambda item: item.chapter_order)
    ]

    return CourseDetail(
        id=course.id,
        title=course.title,
        description=course.description,
        created_at=course.created_at,
        updated_at=course.updated_at,
        thumbnail_url=_course_thumbnail_url(course.id),
        chapters=chapter_briefs,
    )


@router.delete("/{course_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_course(course_id: UUID, db: Session = Depends(get_db)) -> None:
    """Delete an empty course."""
    result = db.execute(
        select(Course)
        .options(selectinload(Course.chapters))
        .where(Course.id == course_id)
    )
    course = result.scalar_one_or_none()
    if course is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Course not found.")
    if course.chapters:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Cannot delete a course that still has chapters.",
        )

    db.delete(course)
    db.commit()
