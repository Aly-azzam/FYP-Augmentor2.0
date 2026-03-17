"""Sequence utilities — smoothing, windowing, and sequence alignment helpers."""


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
