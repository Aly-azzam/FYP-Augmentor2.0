from uuid import UUID

from fastapi import APIRouter, Depends, Query
from sqlalchemy import func
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.models.attempt import Attempt
from app.models.chapter import Chapter
from app.models.course import Course
from app.schemas.evaluation_schema import ProgressOut

router = APIRouter(prefix="/api/progress", tags=["Progress"])


@router.get("", response_model=list[ProgressOut])
async def list_progress(
    user_id: UUID | None = Query(None),
    db: Session = Depends(get_db),
):
    """Get completion-based progress per course."""
    total_rows = (
        db.query(
            Chapter.course_id.label("course_id"),
            func.count(Chapter.id).label("total_chapters"),
        )
        .group_by(Chapter.course_id)
        .all()
    )
    total_map = {str(row.course_id): int(row.total_chapters) for row in total_rows}

    completed_query = (
        db.query(
            Chapter.course_id.label("course_id"),
            func.count(func.distinct(Attempt.chapter_id)).label("completed_chapters"),
        )
        .join(Chapter, Chapter.id == Attempt.chapter_id)
        .filter(Attempt.status.in_(["completed", "evaluated"]))
        .group_by(Chapter.course_id)
    )
    if user_id is not None:
        completed_query = completed_query.filter(Attempt.user_id == str(user_id))
    completed_rows = completed_query.all()
    completed_map = {str(row.course_id): int(row.completed_chapters) for row in completed_rows}

    course_rows = db.query(Course).all()
    result: list[ProgressOut] = []
    for course in course_rows:
        course_id_str = str(course.id)
        total = total_map.get(course_id_str, 0)
        completed = completed_map.get(course_id_str, 0)
        ratio = (float(completed) / float(total)) if total > 0 else 0.0
        result.append(
            ProgressOut(
                course_id=UUID(course_id_str),
                course_title=course.title,
                completed_chapters=completed,
                total_chapters=total,
                completion_ratio=round(ratio, 4),
            )
        )
    return result
