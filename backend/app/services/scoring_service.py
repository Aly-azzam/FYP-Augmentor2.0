"""Scoring helpers for both internal motion scoring and evaluation scoring."""

from __future__ import annotations

from typing import Any, Dict

from app.core.evaluation_constants import (
    ACTIVE_SCORING_METRICS,
    DEVIATION_METRICS,
    DEFAULT_METRIC_WEIGHTS,
    EXCELLENT_SCORE_THRESHOLD,
    FAIR_SCORE_THRESHOLD,
    GOOD_SCORE_THRESHOLD,
    QUALITY_METRICS,
    NEEDS_IMPROVEMENT_SCORE_THRESHOLD,
    REQUIRED_METRICS,
    SCORE_MAX,
    SCORE_MIN,
)
from app.utils.evaluation_utils import clamp, round_metric


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


async def compute_score(metrics: dict[str, Any]) -> int:
    """Compatibility async scoring API returning deterministic numeric score."""
    if not isinstance(metrics, dict):
        raise ValueError("compute_score: metrics must be a dict.")
    return int(compute_final_score(metrics)["score"])


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


def normalize_metric_for_scoring(metric_name: str, raw_value: float) -> float:
    """Convert raw metric value into quality in [0,1]."""
    value = clamp(float(raw_value), 0.0, 1.0)
    if metric_name in DEVIATION_METRICS:
        return clamp(1.0 - value, 0.0, 1.0)
    if metric_name in QUALITY_METRICS:
        return value
    raise ValueError(f"Unknown metric for scoring normalization: {metric_name}")


def compute_metric_breakdown(metrics: dict[str, float]) -> dict[str, dict[str, float]]:
    """Build per-metric scoring breakdown with score-point contributions."""
    missing_metrics = [metric_name for metric_name in REQUIRED_METRICS if metric_name not in metrics]
    if missing_metrics:
        missing = ", ".join(missing_metrics)
        raise ValueError(f"Missing required metrics for scoring: {missing}")

    breakdown: dict[str, dict[str, float]] = {}
    for metric_name in (
        "trajectory_deviation",
        "angle_deviation",
        "velocity_difference",
        "smoothness_score",
        "timing_score",
        "hand_openness_deviation",
        "tool_alignment_deviation",
    ):
        if metric_name not in metrics:
            continue

        raw_value = clamp(float(metrics[metric_name]), 0.0, 1.0)
        if metric_name in DEVIATION_METRICS or metric_name in QUALITY_METRICS:
            normalized_quality = normalize_metric_for_scoring(metric_name, raw_value)
        else:
            normalized_quality = raw_value

        weight = float(DEFAULT_METRIC_WEIGHTS.get(metric_name, 0.0))
        contribution = SCORE_MAX * weight * normalized_quality

        breakdown[metric_name] = {
            "raw_value": round_metric(raw_value),
            "normalized_quality": round_metric(normalized_quality),
            "weight": round_metric(weight),
            "contribution": round_metric(contribution),
        }

    return breakdown


def compute_score_quality(metrics: dict[str, float]) -> float:
    """Compute weighted score quality in [0,1] from active metrics."""
    breakdown = compute_metric_breakdown(metrics)
    total_quality = sum(
        breakdown[metric_name]["weight"] * breakdown[metric_name]["normalized_quality"]
        for metric_name in ACTIVE_SCORING_METRICS
    )
    return round_metric(clamp(total_quality, 0.0, 1.0))


def classify_score(score: int) -> str:
    """Return a simple qualitative label for a numeric score."""
    if score >= EXCELLENT_SCORE_THRESHOLD:
        return "excellent"
    if score >= GOOD_SCORE_THRESHOLD:
        return "good"
    if score >= FAIR_SCORE_THRESHOLD:
        return "fair"
    if score >= NEEDS_IMPROVEMENT_SCORE_THRESHOLD:
        return "needs_improvement"
    return "poor"


def compute_final_score(metrics: dict[str, float]) -> dict[str, Any]:
    """Convert mixed-direction normalized metrics into a final score and label.

    Formula:
    score_quality =
      0.30*trajectory_quality +
      0.25*angle_quality +
      0.20*velocity_quality +
      0.15*smoothness_score +
      0.10*timing_score

    final_score = round(100 * score_quality), clamped to [0,100].
    """
    breakdown = compute_metric_breakdown(metrics)
    total_quality = compute_score_quality(metrics)
    raw_score = SCORE_MAX * total_quality
    final_score = int(round(clamp(raw_score, SCORE_MIN, SCORE_MAX)))
    contribution_sum = round_metric(sum(item["contribution"] for item in breakdown.values()))

    return {
        "score": final_score,
        "score_quality": round_metric(total_quality),
        "label": classify_score(final_score),
        "per_metric_breakdown": breakdown,
        "contribution_sum": contribution_sum,
    }


def compute_numeric_score(metrics: dict[str, float]) -> int:
    """Return only the numeric evaluation score for normalized metrics."""
    return int(compute_final_score(metrics)["score"])
