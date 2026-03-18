"""Compute normalized tool alignment deviation between expert and learner."""

from __future__ import annotations

import math
from typing import Any

from app.utils.evaluation_utils import clamp, round_metric, safe_average

MAX_TOOL_POINT_DISTANCE = math.sqrt(2.0)


def compute_tool_path_difference(
    expert_path: list[list[float]],
    learner_path: list[list[float]],
) -> float:
    """Compute the normalized difference between two tool paths."""
    if not expert_path or not learner_path:
        return 0.0

    shared_length = min(len(expert_path), len(learner_path))
    if shared_length == 0:
        return 0.0

    point_differences: list[float] = []
    for index in range(shared_length):
        expert_point = expert_path[index]
        learner_point = learner_path[index]

        if len(expert_point) < 2 or len(learner_point) < 2:
            raise ValueError("Each tool point must contain at least two values: [x, y].")

        dx = float(expert_point[0]) - float(learner_point[0])
        dy = float(expert_point[1]) - float(learner_point[1])
        distance = math.sqrt((dx * dx) + (dy * dy))
        point_differences.append(clamp(distance / MAX_TOOL_POINT_DISTANCE, 0.0, 1.0))

    return clamp(safe_average(point_differences), 0.0, 1.0)


def compute_tool_alignment_deviation(paired_motion_data: dict[str, Any]) -> float:
    """Compute one normalized tool alignment deviation score from paired motion data."""
    if "expert_motion" not in paired_motion_data or "learner_motion" not in paired_motion_data:
        raise ValueError("Paired motion data must include 'expert_motion' and 'learner_motion'.")

    expert_tool_motion = paired_motion_data["expert_motion"].get("tool_motion")
    learner_tool_motion = paired_motion_data["learner_motion"].get("tool_motion")

    if not isinstance(expert_tool_motion, dict) or not isinstance(learner_tool_motion, dict):
        raise ValueError("'tool_motion' must be a dictionary for both expert and learner.")

    shared_keys = set(expert_tool_motion.keys()) & set(learner_tool_motion.keys())
    if not shared_keys:
        return 0.0

    tool_scores: list[float] = []
    for key in shared_keys:
        expert_path = expert_tool_motion[key]
        learner_path = learner_tool_motion[key]

        if not isinstance(expert_path, list) or not isinstance(learner_path, list):
            raise ValueError(f"Tool motion for '{key}' must be a list.")

        tool_scores.append(compute_tool_path_difference(expert_path, learner_path))

    return round_metric(safe_average(tool_scores))
