"""Compute normalized trajectory deviation between expert and learner."""

from __future__ import annotations

import math
from typing import Any

from app.utils.evaluation_utils import clamp, round_metric, safe_average

MAX_POINT_DISTANCE = math.sqrt(2.0)


def compute_point_distance(expert_point: list[float], learner_point: list[float]) -> float:
    """Compute normalized Euclidean distance between two 2D points."""
    if len(expert_point) < 2 or len(learner_point) < 2:
        raise ValueError("Each trajectory point must contain at least two values: [x, y].")

    dx = float(expert_point[0]) - float(learner_point[0])
    dy = float(expert_point[1]) - float(learner_point[1])
    distance = math.sqrt((dx * dx) + (dy * dy))
    return clamp(distance / MAX_POINT_DISTANCE, 0.0, 1.0)


def compute_trajectory_difference(
    expert_trajectory: list[list[float]],
    learner_trajectory: list[list[float]],
) -> float:
    """Compute the normalized difference between two 2D trajectories."""
    if not expert_trajectory or not learner_trajectory:
        return 0.0

    shared_length = min(len(expert_trajectory), len(learner_trajectory))
    if shared_length == 0:
        return 0.0

    point_distances = [
        compute_point_distance(expert_trajectory[index], learner_trajectory[index])
        for index in range(shared_length)
    ]
    return clamp(safe_average(point_distances), 0.0, 1.0)


def compute_trajectory_deviation(paired_motion_data: dict[str, Any]) -> float:
    """Compute one normalized trajectory deviation score from paired motion data."""
    if "expert_motion" not in paired_motion_data or "learner_motion" not in paired_motion_data:
        raise ValueError("Paired motion data must include 'expert_motion' and 'learner_motion'.")

    expert_trajectories = paired_motion_data["expert_motion"].get("joint_trajectories")
    learner_trajectories = paired_motion_data["learner_motion"].get("joint_trajectories")

    if not isinstance(expert_trajectories, dict) or not isinstance(learner_trajectories, dict):
        raise ValueError("'joint_trajectories' must be a dictionary for both expert and learner.")

    shared_keys = set(expert_trajectories.keys()) & set(learner_trajectories.keys())
    if not shared_keys:
        return 0.0

    trajectory_scores: list[float] = []
    for key in shared_keys:
        expert_trajectory = expert_trajectories[key]
        learner_trajectory = learner_trajectories[key]

        if not isinstance(expert_trajectory, list) or not isinstance(learner_trajectory, list):
            raise ValueError(f"Trajectory series for '{key}' must be a list.")

        trajectory_scores.append(
            compute_trajectory_difference(expert_trajectory, learner_trajectory)
        )

    return round_metric(safe_average(trajectory_scores))
