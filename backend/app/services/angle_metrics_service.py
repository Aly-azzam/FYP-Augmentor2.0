"""Compute normalized angle deviation between expert and learner motion data."""

from __future__ import annotations

from typing import Any

from app.utils.evaluation_utils import clamp, round_metric, safe_average

MAX_ANGLE_DIFFERENCE = 180.0


def compute_series_difference(expert_series: list[float], learner_series: list[float]) -> float:
    """Compute the normalized difference between two angle series.

    The comparison uses only the shared range when lengths differ.
    The returned value is normalized to the range [0.0, 1.0].
    """
    if not expert_series or not learner_series:
        return 0.0

    shared_length = min(len(expert_series), len(learner_series))
    if shared_length == 0:
        return 0.0

    differences = [
        abs(float(expert_series[index]) - float(learner_series[index]))
        for index in range(shared_length)
    ]
    mean_difference = safe_average(differences)
    normalized_difference = mean_difference / MAX_ANGLE_DIFFERENCE
    return clamp(normalized_difference, 0.0, 1.0)


def compute_angle_deviation(paired_motion_data: dict[str, Any]) -> float:
    """Compute one normalized angle deviation score from paired motion data."""
    if "expert_motion" not in paired_motion_data or "learner_motion" not in paired_motion_data:
        raise ValueError("Paired motion data must include 'expert_motion' and 'learner_motion'.")

    expert_angles = paired_motion_data["expert_motion"].get("angle_series")
    learner_angles = paired_motion_data["learner_motion"].get("angle_series")

    if not isinstance(expert_angles, dict) or not isinstance(learner_angles, dict):
        raise ValueError("'angle_series' must be a dictionary for both expert and learner.")

    shared_keys = set(expert_angles.keys()) & set(learner_angles.keys())
    if not shared_keys:
        return 0.0

    series_scores: list[float] = []
    for key in shared_keys:
        expert_series = expert_angles[key]
        learner_series = learner_angles[key]

        if not isinstance(expert_series, list) or not isinstance(learner_series, list):
            raise ValueError(f"Angle series for '{key}' must be a list.")

        series_scores.append(compute_series_difference(expert_series, learner_series))

    return round_metric(safe_average(series_scores))
