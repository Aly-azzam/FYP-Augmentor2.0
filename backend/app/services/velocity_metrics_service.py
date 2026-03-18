"""Compute normalized velocity difference between expert and learner."""

from __future__ import annotations

from typing import Any

from app.utils.evaluation_utils import clamp, round_metric, safe_average

MAX_VELOCITY_DIFFERENCE = 1.0


def compute_velocity_series_difference(
    expert_series: list[float],
    learner_series: list[float],
) -> float:
    """Compute the normalized difference between two velocity series."""
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
    normalized_difference = mean_difference / MAX_VELOCITY_DIFFERENCE
    return clamp(normalized_difference, 0.0, 1.0)


def compute_velocity_difference(paired_motion_data: dict[str, Any]) -> float:
    """Compute one normalized velocity difference score from paired motion data."""
    if "expert_motion" not in paired_motion_data or "learner_motion" not in paired_motion_data:
        raise ValueError("Paired motion data must include 'expert_motion' and 'learner_motion'.")

    expert_profiles = paired_motion_data["expert_motion"].get("velocity_profiles")
    learner_profiles = paired_motion_data["learner_motion"].get("velocity_profiles")

    if not isinstance(expert_profiles, dict) or not isinstance(learner_profiles, dict):
        raise ValueError("'velocity_profiles' must be a dictionary for both expert and learner.")

    shared_keys = set(expert_profiles.keys()) & set(learner_profiles.keys())
    if not shared_keys:
        return 0.0

    velocity_scores: list[float] = []
    for key in shared_keys:
        expert_series = expert_profiles[key]
        learner_series = learner_profiles[key]

        if not isinstance(expert_series, list) or not isinstance(learner_series, list):
            raise ValueError(f"Velocity series for '{key}' must be a list.")

        velocity_scores.append(
            compute_velocity_series_difference(expert_series, learner_series)
        )

    return round_metric(safe_average(velocity_scores))
