from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.orm import Session, selectinload

from app.core.database import get_db
from app.models.chapter import Chapter
from app.models.course import Course
from app.schemas.course_schema import CourseOut, CourseDetail

router = APIRouter(prefix="/api/courses", tags=["Courses"])


@router.get("", response_model=list[CourseOut])
def list_courses(db: Session = Depends(get_db)):
    """List all available courses."""
    result = db.execute(select(Course).order_by(Course.created_at.desc()))
    courses = result.scalars().all()
    return courses


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
        chapters=chapter_briefs,
    )
