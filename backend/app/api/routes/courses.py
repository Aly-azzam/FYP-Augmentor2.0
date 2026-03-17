from uuid import UUID

from fastapi import APIRouter

from app.schemas.course_schema import CourseOut, CourseDetail

router = APIRouter(prefix="/api/courses", tags=["Courses"])


@router.get("", response_model=list[CourseOut])
async def list_courses():
    """List all available courses."""
    # TODO: query real DB
    return []


@router.get("/{course_id}", response_model=CourseDetail)
async def get_course(course_id: UUID):
    """Get course details including chapters."""
    # TODO: query real DB, return 404 if not found
    return CourseDetail(
        id=course_id,
        title="Placeholder Course",
        created_at="2026-01-01T00:00:00Z",
        chapters=[],
    )
