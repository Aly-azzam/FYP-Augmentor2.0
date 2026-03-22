"""Landmark utilities (Perception Engine helpers).

This module contains placeholder helper function skeletons for converting
raw landmark lists into dictionary/structured formats.
"""

from __future__ import annotations

from typing import Any, Dict, Optional


def landmark_list_to_dict(landmarks: Any) -> Dict[str, Any]:
    """Convert a raw landmark list into a structured dictionary (stub)."""
    raise NotImplementedError("Landmark conversion is not implemented yet.")


def format_frame_landmarks(
    frame_index: int,
    timestamp_sec: float,
    left_hand: Optional[Any] = None,
    right_hand: Optional[Any] = None,
) -> Dict[str, Any]:
    """Format per-frame landmarks into a schema-friendly dict (stub)."""
    raise NotImplementedError("Frame landmark formatting is not implemented yet.")
