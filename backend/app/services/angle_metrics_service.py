"""Angle metric helpers for both extraction and pairwise evaluation."""

from __future__ import annotations

import math
from typing import Any, List, Optional

from app.utils.evaluation_utils import clamp, round_metric, safe_average

MAX_ANGLE_DIFFERENCE = 180.0


def calculate_angle(point_a: list[float], point_b: list[float], point_c: list[float]) -> float:
    """Compute the angle ABC (with B as the joint center) in degrees.

    Args:
        point_a: Point A as [x, y, z].
        point_b: Point B (joint center) as [x, y, z].
        point_c: Point C as [x, y, z].

    Returns:
        Angle ABC in degrees.

    Raises:
        ValueError: if any point is malformed or any vector length is zero.
    """

    def _validate_point(p: list[float], name: str) -> List[float]:
        if not isinstance(p, list) or len(p) != 3:
            raise ValueError(f"calculate_angle: {name} must be a list[float] of length 3.")
        try:
            return [float(p[0]), float(p[1]), float(p[2])]
        except (TypeError, ValueError) as exc:
            raise ValueError(f"calculate_angle: {name} contains non-numeric values.") from exc

    a = _validate_point(point_a, "point_a")
    b = _validate_point(point_b, "point_b")
    c = _validate_point(point_c, "point_c")

    # Vectors BA and BC (origin at joint center B)
    ba = (a[0] - b[0], a[1] - b[1], a[2] - b[2])
    bc = (c[0] - b[0], c[1] - b[1], c[2] - b[2])

    ba_len = math.sqrt(ba[0] * ba[0] + ba[1] * ba[1] + ba[2] * ba[2])
    bc_len = math.sqrt(bc[0] * bc[0] + bc[1] * bc[1] + bc[2] * bc[2])

    if ba_len == 0.0:
        raise ValueError("calculate_angle: vector BA length is zero.")
    if bc_len == 0.0:
        raise ValueError("calculate_angle: vector BC length is zero.")

    dot = ba[0] * bc[0] + ba[1] * bc[1] + ba[2] * bc[2]
    cosine = dot / (ba_len * bc_len)
    cosine = max(-1.0, min(1.0, cosine))  # clamp before acos

    return math.degrees(math.acos(cosine))


def compute_hand_joint_angles(hand_landmarks: dict) -> dict:
    """Compute hand joint angles from one hand landmark dictionary.

    Args:
        hand_landmarks: One hand dictionary like:
            {
              "wrist": [x, y, z],
              ...
            }

    Returns:
        Dict with exactly:
        {
          "thumb_ip_angle": ...,
          "index_pip_angle": ...,
          "middle_pip_angle": ...,
          "ring_pip_angle": ...,
          "pinky_pip_angle": ...
        }

    Raises:
        ValueError: if hand_landmarks is None, required landmarks are missing,
            or any required point is malformed.
    """
    if hand_landmarks is None:
        raise ValueError("compute_hand_joint_angles: hand_landmarks must not be None.")
    if not isinstance(hand_landmarks, dict):
        raise ValueError("compute_hand_joint_angles: hand_landmarks must be a dict.")

    def _get_point(name: str) -> list[float]:
        if name not in hand_landmarks:
            raise ValueError(f"compute_hand_joint_angles: missing required landmark '{name}'.")
        point = hand_landmarks[name]
        if not isinstance(point, list) or len(point) != 3:
            raise ValueError(
                f"compute_hand_joint_angles: landmark '{name}' must be a list[float] of length 3."
            )
        try:
            return [float(point[0]), float(point[1]), float(point[2])]
        except (TypeError, ValueError) as exc:
            raise ValueError(f"compute_hand_joint_angles: landmark '{name}' has non-numeric values.") from exc

    thumb_ip_angle = calculate_angle(
        point_a=_get_point("thumb_mcp"),
        point_b=_get_point("thumb_ip"),
        point_c=_get_point("thumb_tip"),
    )

    index_pip_angle = calculate_angle(
        point_a=_get_point("index_finger_mcp"),
        point_b=_get_point("index_finger_pip"),
        point_c=_get_point("index_finger_dip"),
    )

    middle_pip_angle = calculate_angle(
        point_a=_get_point("middle_finger_mcp"),
        point_b=_get_point("middle_finger_pip"),
        point_c=_get_point("middle_finger_dip"),
    )

    ring_pip_angle = calculate_angle(
        point_a=_get_point("ring_finger_mcp"),
        point_b=_get_point("ring_finger_pip"),
        point_c=_get_point("ring_finger_dip"),
    )

    pinky_pip_angle = calculate_angle(
        point_a=_get_point("pinky_mcp"),
        point_b=_get_point("pinky_pip"),
        point_c=_get_point("pinky_dip"),
    )

    return {
        "thumb_ip_angle": thumb_ip_angle,
        "index_pip_angle": index_pip_angle,
        "middle_pip_angle": middle_pip_angle,
        "ring_pip_angle": ring_pip_angle,
        "pinky_pip_angle": pinky_pip_angle,
    }


def compute_frame_hand_angles(frame_data: dict) -> dict:
    """Compute per-hand joint angles for a single processed frame.

    Args:
        frame_data: Dict shaped like:
            {
              "frame_index": ...,
              "timestamp": ...,
              "left_hand": {...} or None,
              "right_hand": {...} or None
            }

    Returns:
        Dict exactly shaped as:
        {
          "frame_index": ...,
          "timestamp": ...,
          "left_hand_angles": {...} or None,
          "right_hand_angles": {...} or None
        }

    Raises:
        ValueError: if required keys are missing or frame_data is malformed.
    """
    if frame_data is None or not isinstance(frame_data, dict):
        raise ValueError("compute_frame_hand_angles: frame_data must be a dict.")

    required_keys = {"frame_index", "timestamp", "left_hand", "right_hand"}
    missing = required_keys - set(frame_data.keys())
    if missing:
        missing_str = ", ".join(sorted(missing))
        raise ValueError(f"compute_frame_hand_angles: frame_data missing keys: {missing_str}.")

    left_hand = frame_data["left_hand"]
    right_hand = frame_data["right_hand"]

    left_hand_angles: Optional[dict]
    right_hand_angles: Optional[dict]

    if left_hand is None:
        left_hand_angles = None
    else:
        if not isinstance(left_hand, dict):
            raise ValueError("compute_frame_hand_angles: left_hand must be a dict or None.")
        left_hand_angles = compute_hand_joint_angles(left_hand)

    if right_hand is None:
        right_hand_angles = None
    else:
        if not isinstance(right_hand, dict):
            raise ValueError("compute_frame_hand_angles: right_hand must be a dict or None.")
        right_hand_angles = compute_hand_joint_angles(right_hand)

    return {
        "frame_index": frame_data["frame_index"],
        "timestamp": frame_data["timestamp"],
        "left_hand_angles": left_hand_angles,
        "right_hand_angles": right_hand_angles,
    }


def compute_video_hand_angles(frames: list[dict]) -> list[dict]:
    """Compute per-frame hand angles for a list of processed frames.

    Args:
        frames: List of processed frame dictionaries.

    Returns:
        List of per-frame angle dictionaries.
    """
    if frames is None or not isinstance(frames, list):
        raise ValueError("compute_video_hand_angles: frames must be a list.")

    return [compute_frame_hand_angles(frame_data=f) for f in frames]


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
