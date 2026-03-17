"""Video utilities — file handling, path generation, format helpers."""

import uuid
from pathlib import Path

from app.core.config import settings


def get_upload_path(filename: str, attempt_id: uuid.UUID) -> Path:
    """Generate storage path for an uploaded practice video."""
    ext = Path(filename).suffix
    dest = settings.STORAGE_ROOT / settings.UPLOAD_DIR / f"{attempt_id}{ext}"
    dest.parent.mkdir(parents=True, exist_ok=True)
    return dest


def get_landmark_path(video_id: str) -> Path:
    """Generate storage path for persisted landmark data."""
    dest = settings.STORAGE_ROOT / settings.PROCESSED_DIR / "landmarks" / f"{video_id}.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    return dest


def get_motion_path(video_id: str) -> Path:
    """Generate storage path for persisted motion representation."""
    dest = settings.STORAGE_ROOT / settings.PROCESSED_DIR / "motion" / f"{video_id}.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    return dest


def get_evaluation_output_path(evaluation_id: uuid.UUID) -> Path:
    """Generate storage path for evaluation result artifact."""
    dest = settings.STORAGE_ROOT / settings.OUTPUT_DIR / "evaluations" / f"{evaluation_id}.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    return dest
