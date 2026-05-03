from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from sqlalchemy import select, text
from sqlalchemy.orm import Session

from app.core.config import settings
from app.core.database import SessionLocal, engine, get_db
from app.db.base import Base
from app.api.routes import courses, chapters, uploads, evaluations, history, progress
from app.api.routes import expert_mediapipe, expert_sam2, inspection, optical_flow
from app.api import sam2_yolo
from app.api.mediapipe import router as mediapipe_router
from app.api.sam2 import router as sam2_router
from fastapi import Depends
from app.models.user import User
from app.models.course import Course
from app.models.chapter import Chapter
from app.models.video import Video
from app.models.attempt import Attempt
from app.models.evaluation import Evaluation
from app.models.evaluation_feedback import EvaluationFeedback


STRAIGHT_LINE_COURSE_ID = "11111111-1111-4111-8111-000000000001"
STRAIGHT_LINE_CHAPTER_ID = "11111111-1111-4111-8111-111111111111"
STRAIGHT_LINE_COURSE_TITLE = "Cut a straight line"
STRAIGHT_LINE_CHAPTER_TITLE = "Straight Line Cutting Expert Demo"



app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(courses.router)
app.include_router(chapters.router)
app.include_router(uploads.router)
app.include_router(evaluations.router)
app.include_router(history.router)
app.include_router(progress.router)
app.include_router(optical_flow.router)
app.include_router(mediapipe_router)
app.include_router(sam2_router)
app.include_router(sam2_yolo.router)
app.include_router(expert_mediapipe.router)
app.include_router(expert_sam2.router)
app.include_router(inspection.router)
app.mount("/storage", StaticFiles(directory=settings.STORAGE_ROOT), name="storage")


@app.on_event("startup")
def ensure_database_tables() -> None:
    """Create core tables if the database is empty or freshly created."""
    Base.metadata.create_all(bind=engine)
    ensure_straight_line_course()


def ensure_straight_line_course() -> None:
    """Ensure the uploadable straight-line cutting chapter exists."""
    session = SessionLocal()
    try:
        course = session.execute(
            select(Course).where(Course.id == STRAIGHT_LINE_COURSE_ID)
        ).scalar_one_or_none()
        if course is None:
            course = Course(
                id=STRAIGHT_LINE_COURSE_ID,
                title=STRAIGHT_LINE_COURSE_TITLE,
                description="Practice controlled straight-line cutting with scissors.",
            )
            session.add(course)
            session.flush()
        else:
            course.title = STRAIGHT_LINE_COURSE_TITLE
            course.description = "Practice controlled straight-line cutting with scissors."

        chapter = session.execute(
            select(Chapter).where(Chapter.id == STRAIGHT_LINE_CHAPTER_ID)
        ).scalar_one_or_none()
        if chapter is None:
            chapter = Chapter(
                id=STRAIGHT_LINE_CHAPTER_ID,
                course_id=STRAIGHT_LINE_COURSE_ID,
                title=STRAIGHT_LINE_CHAPTER_TITLE,
                description="Upload the expert straight-line cutting video for comparison.",
                chapter_order=1,
            )
            session.add(chapter)
        else:
            chapter.course_id = STRAIGHT_LINE_COURSE_ID
            chapter.title = STRAIGHT_LINE_CHAPTER_TITLE
            chapter.description = "Upload the expert straight-line cutting video for comparison."
            chapter.chapter_order = 1

        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


@app.get("/health", tags=["Health"])
async def health_check():
    return {
        "status": "healthy",
        "version": settings.APP_VERSION,
        "pipeline_version": settings.PIPELINE_VERSION,
    }

@app.get("/health/db", tags=["Health"])
def health_db(db: Session = Depends(get_db)):
    db.execute(text("SELECT 1"))
    return {"status": "db connected"}


@app.get("/health/db/smoke", tags=["Health"])
def health_db_smoke(db: Session = Depends(get_db)):
    checks: dict[str, dict[str, object]] = {}
    overall_ok = True

    try:
        db.execute(text("SELECT 1"))
        checks["select_1"] = {"ok": True}
    except Exception as exc:
        checks["select_1"] = {"ok": False, "error": str(exc)}
        overall_ok = False

    model_checks = [
        ("User", User),
        ("Course", Course),
        ("Chapter", Chapter),
        ("Video", Video),
        ("Attempt", Attempt),
        ("Evaluation", Evaluation),
        ("EvaluationFeedback", EvaluationFeedback),
    ]

    for label, model in model_checks:
        try:
            row = db.query(model).first()
            checks[label] = {
                "ok": True,
                "found_row": row is not None,
            }
        except Exception as exc:
            checks[label] = {"ok": False, "error": str(exc)}
            overall_ok = False

    return {
        "status": "db smoke passed" if overall_ok else "db smoke failed",
        "ok": overall_ok,
        "checks": checks,
    }
