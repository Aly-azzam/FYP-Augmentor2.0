from __future__ import annotations

import os
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from fastapi import UploadFile
from sqlalchemy import select
from sqlalchemy.orm import Session, selectinload

from app.core.config import settings
from app.core.constants import ALLOWED_VIDEO_EXTENSIONS, ALLOWED_VIDEO_TYPES, AttemptStatus
from app.models.chapter import Chapter
from app.models.learner_attempt import LearnerAttempt
from app.models.user import User

CHUNK_SIZE_BYTES = 1024 * 1024


class UploadValidationError(ValueError):
    """Raised when the upload payload is invalid."""


class UploadPersistenceError(RuntimeError):
    """Raised when the upload cannot be saved or persisted."""


class RelatedResourceNotFoundError(LookupError):
    """Raised when a required related record is missing."""


@dataclass
class UploadResult:
    attempt: LearnerAttempt
    expert_video_id: Optional[uuid.UUID]


def _get_file_extension(filename: str) -> str:
    return Path(filename).suffix.lower()


def _build_relative_storage_path(attempt_id: uuid.UUID, extension: str) -> Path:
    return Path(settings.UPLOAD_DIR) / settings.LEARNER_UPLOAD_SUBDIR / str(attempt_id) / f"original{extension}"


def validate_upload_file(file: UploadFile) -> str:
    """Validate the uploaded file metadata and return its normalized extension."""
    if file is None:
        raise UploadValidationError("A video file is required.")

    filename = (file.filename or "").strip()
    if not filename:
        raise UploadValidationError("Uploaded file must have a filename.")

    extension = _get_file_extension(filename)
    if extension not in ALLOWED_VIDEO_EXTENSIONS:
        raise UploadValidationError(
            "Unsupported video format. Allowed formats: mp4, mov, avi, webm."
        )

    content_type = (file.content_type or "").strip().lower()
    if content_type and content_type not in ALLOWED_VIDEO_TYPES and content_type != "application/octet-stream":
        raise UploadValidationError(
            "Unsupported content type for uploaded video."
        )

    return extension


async def _get_chapter_with_context(db: Session, chapter_id: uuid.UUID) -> Optional[Chapter]:
    result = db.execute(
        select(Chapter)
        .options(selectinload(Chapter.expert_video))
        .where(Chapter.id == chapter_id)
    )
    return result.scalar_one_or_none()


async def _validate_user_if_provided(db: Session, user_id: Optional[uuid.UUID]) -> None:
    if user_id is None:
        return

    result = db.execute(select(User.id).where(User.id == user_id))
    if result.scalar_one_or_none() is None:
        raise UploadValidationError("Provided user_id does not exist.")


async def save_upload_file(file: UploadFile, relative_path: Path) -> int:
    """Save the uploaded file to disk and return the number of bytes written."""
    destination = settings.STORAGE_ROOT / relative_path
    destination.parent.mkdir(parents=True, exist_ok=True)

    bytes_written = 0
    try:
        with destination.open("wb") as output:
            while chunk := await file.read(CHUNK_SIZE_BYTES):
                bytes_written += len(chunk)
                if bytes_written > settings.MAX_UPLOAD_SIZE_BYTES:
                    raise UploadValidationError(
                        f"Uploaded video exceeds the {settings.MAX_UPLOAD_SIZE_BYTES} byte size limit."
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
        raise UploadValidationError("Uploaded video file is empty.")

    return bytes_written


async def create_learner_attempt_upload(
    db: Session,
    *,
    chapter_id: uuid.UUID,
    file: UploadFile,
    user_id: Optional[uuid.UUID] = None,
) -> UploadResult:
    """Validate, save, and persist a learner upload for a chapter."""
    extension = validate_upload_file(file)

    chapter = await _get_chapter_with_context(db, chapter_id)
    if chapter is None:
        raise RelatedResourceNotFoundError("Chapter not found.")

    await _validate_user_if_provided(db, user_id)

    attempt_id = uuid.uuid4()
    relative_path = _build_relative_storage_path(attempt_id, extension)

    attempt = LearnerAttempt(
        id=attempt_id,
        chapter_id=chapter.id,
        user_id=user_id,
        status=AttemptStatus.CREATED.value,
        original_filename=file.filename,
        content_type=(file.content_type or None),
        video_path=str(relative_path).replace(os.sep, "/"),
    )

    try:
        file_size_bytes = await save_upload_file(file, relative_path)
        attempt.file_size_bytes = file_size_bytes
        attempt.status = AttemptStatus.UPLOADED.value

        db.add(attempt)
        db.commit()
        db.refresh(attempt)
    except UploadValidationError:
        db.rollback()
        raise
    except Exception as exc:
        db.rollback()
        saved_file = settings.STORAGE_ROOT / relative_path
        if saved_file.exists():
            saved_file.unlink()
        raise UploadPersistenceError("Failed to save learner video upload.") from exc

    expert_video_id = chapter.expert_video.id if chapter.expert_video is not None else None
    return UploadResult(attempt=attempt, expert_video_id=expert_video_id)
