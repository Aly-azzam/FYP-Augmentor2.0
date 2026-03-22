"""Trajectory Metrics Service: simple spatial trajectory metrics.

This module computes minimal, safe trajectory metrics from extracted hand
landmark dictionaries. It does not compare expert vs learner, and it does
not implement DTW, scoring, velocity, or any aggregation beyond per-video
hand totals.
"""

from __future__ import annotations

import math
from typing import Dict, List


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
