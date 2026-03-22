"""Shared constants for the AugMentor backend.

This module also includes placeholders used by the Perception Engine module
until real inference and video handling logic is implemented.
"""

from enum import Enum


class AttemptStatus(str, Enum):
    CREATED = "created"
    UPLOADED = "uploaded"
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    COMPLETED_WITH_WARNINGS = "completed_with_warnings"
    FAILED = "failed"


ALLOWED_TRANSITIONS: dict[AttemptStatus, list[AttemptStatus]] = {
    AttemptStatus.CREATED: [AttemptStatus.UPLOADED],
    AttemptStatus.UPLOADED: [AttemptStatus.QUEUED, AttemptStatus.FAILED],
    AttemptStatus.QUEUED: [AttemptStatus.RUNNING, AttemptStatus.FAILED],
    AttemptStatus.RUNNING: [
        AttemptStatus.COMPLETED,
        AttemptStatus.COMPLETED_WITH_WARNINGS,
        AttemptStatus.FAILED,
    ],
    AttemptStatus.COMPLETED: [],
    AttemptStatus.COMPLETED_WITH_WARNINGS: [],
    AttemptStatus.FAILED: [],
}

ALLOWED_VIDEO_TYPES = {"video/mp4", "video/quicktime"}
MAX_VIDEO_DURATION_SECONDS = 120

# Perception Engine placeholders (used by stubs in the perception pipeline)
SUPPORTED_VIDEO_FORMATS: set[str] = {".mp4", ".mov", ".mkv", ".avi"}
DEFAULT_FPS: float = 30.0
