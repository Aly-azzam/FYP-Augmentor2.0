"""Video utilities (Perception Engine helpers).

This module contains small helper function skeletons used by the
Perception Engine for working with video filenames and frame timestamps.
"""

from __future__ import annotations


def is_supported_video_format(filename: str) -> bool:
    """Check whether the filename's extension is supported (stub).

    Placeholder stub; later this will validate based on configured formats.
    """
    raise NotImplementedError("Video format validation is not implemented yet.")


def frame_to_timestamp(frame_index: int, fps: float) -> float:
    """Convert a frame index into a timestamp in seconds (stub)."""
    raise NotImplementedError("Frame-to-timestamp conversion is not implemented yet.")
