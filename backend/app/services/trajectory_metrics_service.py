"""Trajectory metric helpers for both extraction and pairwise evaluation."""

from __future__ import annotations

import math
from typing import Any, List

from app.utils.evaluation_utils import clamp, round_metric, safe_average

MAX_POINT_DISTANCE = math.sqrt(2.0)


def euclidean_distance(point_a: list[float], point_b: list[float]) -> float:
    """Compute standard 3D Euclidean distance between two points.

    Args:
        point_a: [x, y, z]
        point_b: [x, y, z]

    Returns:
        Distance in the same units as the input coordinates.

    Raises:
        ValueError: if either point is malformed.
    """

    def _validate_point(p: list[float], name: str) -> List[float]:
        if not isinstance(p, list) or len(p) != 3:
            raise ValueError(f"euclidean_distance: {name} must be a list[float] of length 3.")
        try:
            return [float(p[0]), float(p[1]), float(p[2])]
        except (TypeError, ValueError) as exc:
            raise ValueError(f"euclidean_distance: {name} contains non-numeric values.") from exc

    a = _validate_point(point_a, "point_a")
    b = _validate_point(point_b, "point_b")

    dx = a[0] - b[0]
    dy = a[1] - b[1]
    dz = a[2] - b[2]
    return math.sqrt(dx * dx + dy * dy + dz * dz)


def get_hand_reference_point(hand_landmarks: dict) -> list[float]:
    """Extract the wrist reference point from a hand landmark dictionary.

    Args:
        hand_landmarks: One hand dict like {"wrist": [x, y, z], ...}

    Returns:
        The wrist point as [x, y, z].

    Raises:
        ValueError: if hand_landmarks is None, or if wrist is missing/malformed.
    """
    if hand_landmarks is None:
        raise ValueError("get_hand_reference_point: hand_landmarks must not be None.")
    if not isinstance(hand_landmarks, dict):
        raise ValueError("get_hand_reference_point: hand_landmarks must be a dict.")

    if "wrist" not in hand_landmarks:
        raise ValueError("get_hand_reference_point: missing required 'wrist' landmark.")

    wrist = hand_landmarks["wrist"]
    if not isinstance(wrist, list) or len(wrist) != 3:
        raise ValueError("get_hand_reference_point: 'wrist' must be a list[float] of length 3.")

    try:
        return [float(wrist[0]), float(wrist[1]), float(wrist[2])]
    except (TypeError, ValueError) as exc:
        raise ValueError("get_hand_reference_point: 'wrist' contains non-numeric values.") from exc


def compute_frame_displacement(previous_hand: dict, current_hand: dict) -> float:
    """Compute the frame-to-frame wrist displacement between two hands."""
    prev = get_hand_reference_point(previous_hand)
    curr = get_hand_reference_point(current_hand)
    return euclidean_distance(prev, curr)


def compute_hand_trajectory_metrics(frames: list[dict], hand_key: str) -> dict:
    """Compute simple trajectory metrics for one hand across frames.

    Args:
        frames: Processed frame dicts shaped like:
            {
              "frame_index": ...,
              "timestamp": ...,
              "left_hand": {...} or None,
              "right_hand": {...} or None
            }
        hand_key: Either "left_hand" or "right_hand".

    Returns:
        Dict exactly shaped as:
        {
          "hand_key": hand_key,
          "total_displacement": ...,
          "valid_frame_count": ...,
          "displacement_series": [...]
        }

    Raises:
        ValueError: if input structure is invalid.
    """
    if hand_key not in ("left_hand", "right_hand"):
        raise ValueError("compute_hand_trajectory_metrics: hand_key must be 'left_hand' or 'right_hand'.")

    if frames is None or not isinstance(frames, list):
        raise ValueError("compute_hand_trajectory_metrics: frames must be a list of dicts.")

    required_keys = {"frame_index", "timestamp", "left_hand", "right_hand"}

    valid_frame_count = 0
    displacement_series: List[float] = []
    total_displacement = 0.0

    prev_hand: dict | None = None

    for i, frame in enumerate(frames):
        if not isinstance(frame, dict):
            raise ValueError(f"compute_hand_trajectory_metrics: frames[{i}] must be a dict.")

        missing = required_keys - set(frame.keys())
        if missing:
            missing_str = ", ".join(sorted(missing))
            raise ValueError(f"compute_hand_trajectory_metrics: frames[{i}] missing keys: {missing_str}.")

        current_hand = frame.get(hand_key)
        if current_hand is None:
            continue

        if not isinstance(current_hand, dict):
            raise ValueError(f"compute_hand_trajectory_metrics: frames[{i}].{hand_key} must be a dict or None.")

        valid_frame_count += 1
        if prev_hand is not None:
            step_disp = compute_frame_displacement(prev_hand, current_hand)
            displacement_series.append(step_disp)
            total_displacement += step_disp

        prev_hand = current_hand

    if valid_frame_count < 2:
        return {
            "hand_key": hand_key,
            "total_displacement": 0.0,
            "valid_frame_count": valid_frame_count,
            "displacement_series": [],
        }

    return {
        "hand_key": hand_key,
        "total_displacement": total_displacement,
        "valid_frame_count": valid_frame_count,
        "displacement_series": displacement_series,
    }


def compute_video_trajectory_metrics(frames: list[dict]) -> dict:
    """Compute trajectory metrics for both hands for a video."""
    if frames is None or not isinstance(frames, list):
        raise ValueError("compute_video_trajectory_metrics: frames must be a list of dicts.")

    return {
        "left_hand": compute_hand_trajectory_metrics(frames, "left_hand"),
        "right_hand": compute_hand_trajectory_metrics(frames, "right_hand"),
    }


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
