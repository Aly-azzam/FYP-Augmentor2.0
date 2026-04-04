from uuid import UUID

from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.models.attempt import Attempt
from app.models.chapter import Chapter
from app.models.course import Course
from app.models.evaluation import Evaluation
from app.schemas.evaluation_schema import HistoryEntry

router = APIRouter(prefix="/api/history", tags=["History"])


@router.get("", response_model=list[HistoryEntry])
async def list_history(
    user_id: UUID | None = Query(None),
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    db: Session = Depends(get_db),
):
    """Get evaluation history derived from LearnerAttempt + EvaluationResult."""
    query = (
        db.query(
            Attempt.id.label("attempt_id"),
            Attempt.chapter_id.label("chapter_id"),
            Attempt.status.label("attempt_status"),
            Attempt.created_at.label("attempt_created_at"),
            Chapter.course_id.label("course_id"),
            Chapter.title.label("chapter_title"),
            Course.title.label("course_title"),
            Evaluation.overall_score.label("overall_score"),
            Evaluation.status.label("evaluation_status"),
        )
        .join(Chapter, Chapter.id == Attempt.chapter_id)
        .join(Course, Course.id == Chapter.course_id)
        .outerjoin(Evaluation, Evaluation.attempt_id == Attempt.id)
        .order_by(Attempt.created_at.desc())
        .limit(limit)
        .offset(offset)
    )
    if user_id is not None:
        query = query.filter(Attempt.user_id == str(user_id))

    rows = query.all()
    results: list[HistoryEntry] = []
    for row in rows:
        status_value = str(row.evaluation_status or row.attempt_status)
        score_value = None
        if row.overall_score is not None:
            score_value = int(round(float(row.overall_score)))

        results.append(
            HistoryEntry(
                attempt_id=UUID(str(row.attempt_id)),
                chapter_id=UUID(str(row.chapter_id)),
                course_id=UUID(str(row.course_id)),
                course_title=str(row.course_title),
                chapter_title=str(row.chapter_title),
                status=status_value,
                score=score_value,
                created_at=row.attempt_created_at,
            )
        )

    return results
