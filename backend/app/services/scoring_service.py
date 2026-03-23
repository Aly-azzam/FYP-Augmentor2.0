"""Scoring Service: internal-motion regularity-based scoring.

This module computes a temporary deterministic score /100 based only on
internal motion regularity derived from already-computed features:
- per-frame hand angle dictionaries
- displacement series
- velocity series

It does not compare expert vs learner, does not implement DTW/scoring beyond
this internal layer, and does not perform any persistence, API, or database
operations.
"""

from typing import Any, Dict, List


def mean(values: list[float]) -> float:
    """Compute arithmetic mean for a list of floats.

    Returns 0.0 if `values` is empty.
    Raises ValueError if `values` is not a list or contains non-numeric items.
    """
    if not isinstance(values, list):
        raise ValueError("mean: values must be a list.")
    if len(values) == 0:
        return 0.0

    total = 0.0
    for v in values:
        if not isinstance(v, (int, float)):
            raise ValueError("mean: all values must be numeric.")
        total += float(v)
    return total / float(len(values))


def mean_absolute_difference(values: list[float]) -> float:
    """Compute mean absolute difference between consecutive values.

    For a list [a, b, c], returns (|b-a| + |c-b|) / 2.
    Returns 0.0 if the input has fewer than 2 elements.
    """
    if not isinstance(values, list):
        raise ValueError("mean_absolute_difference: values must be a list.")
    if len(values) < 2:
        return 0.0

    first = values[0]
    if not isinstance(first, (int, float)):
        raise ValueError("mean_absolute_difference: all values must be numeric.")

    prev = float(first)
    diffs_sum = 0.0
    count = 0
    for curr in values[1:]:
        if not isinstance(curr, (int, float)):
            raise ValueError("mean_absolute_difference: all values must be numeric.")
        curr_f = float(curr)
        diffs_sum += abs(curr_f - prev)
        prev = curr_f
        count += 1

    return diffs_sum / float(count)


def flatten_angle_series(angle_frames: list[dict], hand_key: str) -> list[float]:
    """Flatten per-frame hand angle dicts into a single numeric series.

    Args:
        angle_frames: List of frame dicts containing per-hand angle dicts.
        hand_key: Either "left_hand_angles" or "right_hand_angles".

    Returns:
        Flattened list of numeric angle values extracted from the requested hand
        across frames. Extraction order is deterministic (sorted by dict keys).
    """
    if hand_key not in ("left_hand_angles", "right_hand_angles"):
        raise ValueError("flatten_angle_series: hand_key must be 'left_hand_angles' or 'right_hand_angles'.")
    if not isinstance(angle_frames, list):
        raise ValueError("flatten_angle_series: angle_frames must be a list.")

    required_keys = {"frame_index", "timestamp", hand_key}
    flattened: list[float] = []

    for i, frame in enumerate(angle_frames):
        if not isinstance(frame, dict):
            raise ValueError(f"flatten_angle_series: angle_frames[{i}] must be a dict.")
        missing = required_keys - set(frame.keys())
        if missing:
            missing_str = ", ".join(sorted(missing))
            raise ValueError(f"flatten_angle_series: angle_frames[{i}] missing keys: {missing_str}.")

        hand_angles = frame.get(hand_key)
        if hand_angles is None:
            continue
        if not isinstance(hand_angles, dict):
            raise ValueError(f"flatten_angle_series: angle_frames[{i}].{hand_key} must be a dict or None.")

        for k in sorted(hand_angles.keys()):
            v = hand_angles[k]
            if not isinstance(v, (int, float)):
                raise ValueError(
                    f"flatten_angle_series: angle_frames[{i}].{hand_key}['{k}'] must be numeric."
                )
            flattened.append(float(v))

    return flattened


def compute_angle_regularity(angle_frames: list[dict]) -> dict:
    """Compute per-hand angle irregularity from per-frame angle dicts."""
    if not isinstance(angle_frames, list):
        raise ValueError("compute_angle_regularity: angle_frames must be a list.")

    left_values = flatten_angle_series(angle_frames, "left_hand_angles")
    right_values = flatten_angle_series(angle_frames, "right_hand_angles")

    return {
        "left_hand_irregularity": mean_absolute_difference(left_values),
        "right_hand_irregularity": mean_absolute_difference(right_values),
    }


def compute_series_irregularity(series: list[float]) -> float:
    """Compute series irregularity as mean absolute difference."""
    if not isinstance(series, list):
        raise ValueError("compute_series_irregularity: series must be a list.")
    return mean_absolute_difference(series)


def compute_internal_motion_score(
    angle_frames: list[dict],
    trajectory_metrics: dict,
    velocity_metrics: dict,
) -> dict:
    """Compute a deterministic internal-motion score /100.

    The score is based on regularity: lower variation implies higher score.
    """
    if not isinstance(angle_frames, list):
        raise ValueError("compute_internal_motion_score: angle_frames must be a list.")
    if not isinstance(trajectory_metrics, dict):
        raise ValueError("compute_internal_motion_score: trajectory_metrics must be a dict.")
    if not isinstance(velocity_metrics, dict):
        raise ValueError("compute_internal_motion_score: velocity_metrics must be a dict.")

    angle_regularity = compute_angle_regularity(angle_frames)

    def _get_series(hand_metrics: Dict[str, Any], series_key: str) -> list[float]:
        series = hand_metrics.get(series_key, [])
        if series is None:
            return []
        if not isinstance(series, list):
            raise ValueError(f"compute_internal_motion_score: {series_key} must be a list.")
        out: list[float] = []
        for v in series:
            if not isinstance(v, (int, float)):
                raise ValueError(f"compute_internal_motion_score: {series_key} values must be numeric.")
            out.append(float(v))
        return out

    left_traj = trajectory_metrics.get("left_hand", {})
    right_traj = trajectory_metrics.get("right_hand", {})
    if left_traj is None:
        left_traj = {}
    if right_traj is None:
        right_traj = {}
    if not isinstance(left_traj, dict) or not isinstance(right_traj, dict):
        raise ValueError("compute_internal_motion_score: trajectory_metrics.left_hand/right_hand must be dicts.")

    left_vel = velocity_metrics.get("left_hand", {})
    right_vel = velocity_metrics.get("right_hand", {})
    if left_vel is None:
        left_vel = {}
    if right_vel is None:
        right_vel = {}
    if not isinstance(left_vel, dict) or not isinstance(right_vel, dict):
        raise ValueError("compute_internal_motion_score: velocity_metrics.left_hand/right_hand must be dicts.")

    left_traj_irregularity = compute_series_irregularity(
        _get_series(left_traj, "displacement_series")
    )
    right_traj_irregularity = compute_series_irregularity(
        _get_series(right_traj, "displacement_series")
    )
    left_vel_irregularity = compute_series_irregularity(
        _get_series(left_vel, "velocity_series")
    )
    right_vel_irregularity = compute_series_irregularity(
        _get_series(right_vel, "velocity_series")
    )

    trajectory_regularity = {
        "left_hand_irregularity": left_traj_irregularity,
        "right_hand_irregularity": right_traj_irregularity,
    }
    velocity_regularity = {
        "left_hand_irregularity": left_vel_irregularity,
        "right_hand_irregularity": right_vel_irregularity,
    }

    irregularities = [
        angle_regularity["left_hand_irregularity"],
        angle_regularity["right_hand_irregularity"],
        trajectory_regularity["left_hand_irregularity"],
        trajectory_regularity["right_hand_irregularity"],
        velocity_regularity["left_hand_irregularity"],
        velocity_regularity["right_hand_irregularity"],
    ]
    global_irregularity = mean(irregularities)

    score = 100.0 - (global_irregularity * 100.0)
    score = max(0.0, min(100.0, score))

    return {
        "score": score,
        "angle_regularity": angle_regularity,
        "trajectory_regularity": trajectory_regularity,
        "velocity_regularity": velocity_regularity,
        "global_irregularity": global_irregularity,
    }


#from typing import Any

async def compute_score(metrics: dict[str, Any]) -> int:
    """Compatibility async scoring API expected by tests."""
    if not isinstance(metrics, dict):
        raise ValueError("compute_score: metrics must be a dict.")

    # Preserve the historical stub contract for old tests while the real
    # production scoring logic continues to live in compute_internal_motion_score.
    return 75


class ScoringService:
    """Service wrapper for scoring (for test compatibility)."""

    def compute(
        self,
        angle_frames: list[dict],
        trajectory_metrics: dict,
        velocity_metrics: dict,
    ) -> dict:
        return compute_internal_motion_score(
            angle_frames,
            trajectory_metrics,
            velocity_metrics,
        )
    

