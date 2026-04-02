from sqlalchemy import select

from app.core.database import SessionLocal
from app.models.chapter import Chapter
from app.models.course import Course
from app.models.expert_video import ExpertVideo
from app.services.media_service import find_first_expert_video_file, normalize_storage_key

COURSE_TITLE = "MVP Upload Demo Course"
CHAPTER_TITLE = "MVP Upload Demo Chapter"


def main() -> None:
    expert_file = find_first_expert_video_file()
    expert_storage_key = normalize_storage_key(expert_file) if expert_file is not None else None

    session = SessionLocal()
    try:
        course_result = session.execute(
            select(Course).where(Course.title == COURSE_TITLE)
        )
        course = course_result.scalar_one_or_none()
        if course is None:
            course = Course(
                title=COURSE_TITLE,
                description="Seeded course for testing the learner upload flow.",
                instructor="AugMentor Team",
                difficulty="beginner",
            )
            session.add(course)
            session.flush()

        chapter_result = session.execute(
            select(Chapter).where(
                Chapter.course_id == course.id,
                Chapter.title == CHAPTER_TITLE,
            )
        )
        chapter = chapter_result.scalar_one_or_none()
        if chapter is None:
            chapter = Chapter(
                course_id=course.id,
                title=CHAPTER_TITLE,
                order=1,
            )
            session.add(chapter)
            session.flush()

        expert_video_result = session.execute(
            select(ExpertVideo).where(ExpertVideo.chapter_id == chapter.id)
        )
        expert_video = expert_video_result.scalar_one_or_none()
        if expert_storage_key is not None and expert_video is None:
            expert_video = ExpertVideo(
                chapter_id=chapter.id,
                file_path=expert_storage_key,
            )
            session.add(expert_video)
            session.flush()
        elif expert_storage_key is not None and expert_video is not None:
            expert_video.file_path = expert_storage_key

        session.commit()

        print("Seed data ready:")
        print(f"course_id={course.id}")
        print(f"chapter_id={chapter.id}")
        print(f"expert_video_id={expert_video.id if expert_video else 'created on commit'}")
        if expert_storage_key is not None:
            print(f"expert_video_storage_key={expert_storage_key}")
        else:
            print("warning=no expert video file found under backend/storage/expert")
    finally:
        session.close()


if __name__ == "__main__":
    main()
