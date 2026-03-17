"""Preprocessing Service — video validation and preparation.

Responsibilities:
- validate video format and duration
- extract basic metadata (fps, resolution, duration)
- prepare video for downstream perception
"""

from pathlib import Path
from typing import Any


async def validate_video(file_path: Path) -> dict[str, Any]:
    """Validate uploaded video file.

    Returns dict with: valid (bool), fps, duration_seconds, resolution, error (if any).
    """
    # TODO: implement real validation with ffprobe or opencv
    return {
        "valid": True,
        "fps": 30.0,
        "duration_seconds": 15.0,
        "resolution": (1920, 1080),
        "error": None,
    }


async def extract_metadata(file_path: Path) -> dict[str, Any]:
    """Extract video metadata for storage."""
    # TODO: implement with ffprobe or opencv
    return {
        "fps": 30.0,
        "duration_seconds": 15.0,
        "total_frames": 450,
        "resolution": (1920, 1080),
    }
