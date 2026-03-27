from __future__ import annotations

from app.core.database import Base, SessionLocal, engine
from app.models.chapter import Chapter
from app.models.course import Course
from app.models.evaluation_result import EvaluationResult
from app.models.expert_video import ExpertVideo
from app.models.learner_attempt import LearnerAttempt
from app.models.user import User


def test_database_insert_and_query_evaluation_result() -> None:
    _ = ExpertVideo
    _ = User
    Base.metadata.drop_all(bind=engine)
    Base.metadata.create_all(bind=engine)
    session = SessionLocal()
    try:
        course = Course(title="DB Test Course")
        session.add(course)
        session.flush()

        chapter = Chapter(course_id=course.id, title="DB Test Chapter", order=1)
        session.add(chapter)
        session.flush()

        attempt = LearnerAttempt(
            chapter_id=chapter.id,
            status="created",
        )
        session.add(attempt)
        session.flush()

        inserted = EvaluationResult(
            attempt_id=attempt.id,
            score=88,
        )
        session.add(inserted)
        session.commit()

        fetched = (
            session.query(EvaluationResult)
            .filter(EvaluationResult.attempt_id == attempt.id)
            .first()
        )
        assert fetched is not None
        assert fetched.score == 88
    finally:
        session.close()

