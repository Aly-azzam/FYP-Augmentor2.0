"""Landmark utilities — filtering, confidence checks, interpolation helpers."""

from typing import Optional


def filter_low_confidence(
    landmarks: list[dict],
    min_confidence: float = 0.5,
) -> list[dict]:
    """Filter out landmarks below confidence threshold."""
    return [lm for lm in landmarks if (lm.get("confidence") or 0) >= min_confidence]


def interpolate_missing_frames(
    frames: list[Optional[dict]],
    max_gap: int = 5,
) -> list[Optional[dict]]:
    """Interpolate missing frames up to max_gap consecutive frames.

    Returns list with interpolated values where possible, None otherwise.
    """
    # TODO: implement linear interpolation for short gaps
    return frames


def count_usable_frames(frames: list[Optional[dict]]) -> int:
    """Count frames that have at least one hand detected."""
    return sum(1 for f in frames if f is not None)
