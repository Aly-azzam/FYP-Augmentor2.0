"""Sequence utilities — smoothing, windowing, and sequence alignment helpers."""

from collections.abc import Iterable

from app.core.motion_constants import DEFAULT_COORD_DIM
from app.utils.math_utils import zero_vector


def compute_dt(fps: float) -> float:
    """Compute frame delta time from FPS."""
    if fps <= 0:
        raise ValueError("fps must be greater than zero")
    return 1.0 / fps


def landmark_to_vector(landmark: dict | None, dim: int = DEFAULT_COORD_DIM) -> list[float]:
    """Convert a landmark dict into a uniform [x, y, z] vector.

    Missing landmarks return a zero vector to keep the sequence deterministic.
    """
    if not landmark:
        return zero_vector(dim)

    return [
        float(landmark.get("x", 0.0)),
        float(landmark.get("y", 0.0)),
        float(landmark.get("z", 0.0)),
    ][:dim]


def trim_empty_frames(frames: list[dict]) -> list[dict]:
    """Trim leading and trailing frames with no non-zero positions."""
    if not frames:
        return []

    def has_signal(frame: dict) -> bool:
        positions = frame.get("positions", {})
        return any(any(value != 0.0 for value in vector) for vector in positions.values())

    start = 0
    end = len(frames)

    while start < end and not has_signal(frames[start]):
        start += 1
    while end > start and not has_signal(frames[end - 1]):
        end -= 1

    return frames[start:end]


def smooth_sequence(values: list[float], window: int = 3) -> list[float]:
    """Simple moving average smoothing.

    Returns smoothed list of same length (edges use smaller windows).
    """
    if len(values) <= window:
        return values
    result = []
    for i in range(len(values)):
        start = max(0, i - window // 2)
        end = min(len(values), i + window // 2 + 1)
        result.append(sum(values[start:end]) / (end - start))
    return result


def compute_time_deltas(timestamps: list[float]) -> list[float]:
    """Compute per-frame time deltas from timestamps in seconds."""
    if len(timestamps) < 2:
        return []
    return [timestamps[i] - timestamps[i - 1] for i in range(1, len(timestamps))]
