"""Velocity metric helpers for both extraction and pairwise evaluation."""

from __future__ import annotations

from typing import Any, List, Union

from app.utils.evaluation_utils import clamp, round_metric, safe_average

MAX_VELOCITY_DIFFERENCE = 1.0


Number = Union[int, float]


def compute_velocity(displacement: float, delta_time: float) -> float:
    """Compute scalar velocity from displacement and time delta.

    Args:
        displacement: Non-negative scalar displacement.
        delta_time: Time delta in seconds (must be > 0).

    Returns:
        displacement / delta_time

    Raises:
        ValueError: if displacement is invalid or delta_time is invalid.
    """
    if not isinstance(displacement, (int, float)):
        raise ValueError("compute_velocity: displacement must be numeric.")
    if displacement < 0:
        raise ValueError("compute_velocity: displacement must be >= 0.")
    if not isinstance(delta_time, (int, float)):
        raise ValueError("compute_velocity: delta_time must be numeric.")
    if delta_time <= 0:
        raise ValueError("compute_velocity: delta_time must be > 0.")
    return float(displacement) / float(delta_time)


def get_hand_reference_point(hand_landmarks: dict) -> list[float]:
    """Extract the reference point (wrist) from a hand landmark dict."""
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


def euclidean_distance(point_a: list[float], point_b: list[float]) -> float:
    """Compute 3D Euclidean distance between two [x, y, z] points."""
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
    return (dx * dx + dy * dy + dz * dz) ** 0.5


def compute_hand_velocity_metrics(frames: list[dict], hand_key: str) -> dict:
    """Compute per-step hand velocities and mean velocity across a video.

    Notes:
        - Only consecutive frames where the selected hand exists are used.
        - If delta_time <= 0 for a pair, that pair is skipped (no crash).
        - Timestamps are still validated to be numeric and present.
    """
    if hand_key not in ("left_hand", "right_hand"):
        raise ValueError("compute_hand_velocity_metrics: hand_key must be 'left_hand' or 'right_hand'.")
    if frames is None or not isinstance(frames, list):
        raise ValueError("compute_hand_velocity_metrics: frames must be a list of dicts.")

    required_frame_keys = {"frame_index", "timestamp", "left_hand", "right_hand"}

    valid_frame_count = 0
    velocity_series: List[float] = []

    prev_hand: dict | None = None
    prev_timestamp: float | None = None

    for i, frame in enumerate(frames):
        if not isinstance(frame, dict):
            raise ValueError(f"compute_hand_velocity_metrics: frames[{i}] must be a dict.")

        missing = required_frame_keys - set(frame.keys())
        if missing:
            missing_str = ", ".join(sorted(missing))
            raise ValueError(f"compute_hand_velocity_metrics: frames[{i}] missing keys: {missing_str}.")

        timestamp = frame["timestamp"]
        if not isinstance(timestamp, (int, float)):
            raise ValueError(
                f"compute_hand_velocity_metrics: frames[{i}].timestamp must be numeric; got {type(timestamp).__name__}."
            )
        timestamp_f = float(timestamp)

        current_hand = frame.get(hand_key)
        if current_hand is None:
            continue
        if not isinstance(current_hand, dict):
            raise ValueError(f"compute_hand_velocity_metrics: frames[{i}].{hand_key} must be a dict or None.")

        valid_frame_count += 1

        if prev_hand is not None and prev_timestamp is not None:
            displacement = euclidean_distance(
                get_hand_reference_point(prev_hand),
                get_hand_reference_point(current_hand),
            )
            delta_time = timestamp_f - prev_timestamp

            if delta_time > 0:
                velocity_series.append(compute_velocity(displacement=displacement, delta_time=delta_time))

        prev_hand = current_hand
        prev_timestamp = timestamp_f

    if not velocity_series:
        mean_velocity = 0.0
    else:
        mean_velocity = sum(velocity_series) / len(velocity_series)

    return {
        "hand_key": hand_key,
        "velocity_series": velocity_series,
        "mean_velocity": mean_velocity,
        "valid_frame_count": valid_frame_count,
    }


def compute_video_velocity_metrics(frames: list[dict]) -> dict:
    """Compute velocity metrics for both hands for a video."""
    if frames is None or not isinstance(frames, list):
        raise ValueError("compute_video_velocity_metrics: frames must be a list of dicts.")

    return {
        "left_hand": compute_hand_velocity_metrics(frames, "left_hand"),
        "right_hand": compute_hand_velocity_metrics(frames, "right_hand"),
    }


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
    modern_channels = paired_motion_data.get("modern_aligned_channels")
    if isinstance(modern_channels, dict):
        channel_map = modern_channels.get("velocity_channels")
        if isinstance(channel_map, dict) and channel_map:
            velocity_scores: list[float] = []
            for channel_payload in channel_map.values():
                if not isinstance(channel_payload, dict):
                    continue
                expert_series = channel_payload.get("expert", [])
                learner_series = channel_payload.get("learner", [])
                if not isinstance(expert_series, list) or not isinstance(learner_series, list):
                    continue
                velocity_scores.append(compute_velocity_series_difference(expert_series, learner_series))
            if velocity_scores:
                return round_metric(safe_average(velocity_scores))

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


def compute_smoothness_score(paired_motion_data: dict[str, Any]) -> float:
    """Compute smoothness quality [0,1] from aligned velocity-channel jerk proxies."""
    modern_channels = paired_motion_data.get("modern_aligned_channels")
    if not isinstance(modern_channels, dict):
        return 1.0

    channel_map = modern_channels.get("velocity_channels")
    if not isinstance(channel_map, dict) or not channel_map:
        return 1.0

    channel_differences: list[float] = []
    for channel_payload in channel_map.values():
        if not isinstance(channel_payload, dict):
            continue
        expert_series = channel_payload.get("expert", [])
        learner_series = channel_payload.get("learner", [])
        if not isinstance(expert_series, list) or not isinstance(learner_series, list):
            continue
        expert_jerk = _jerk_proxy_series(expert_series)
        learner_jerk = _jerk_proxy_series(learner_series)
        shared_length = min(len(expert_jerk), len(learner_jerk))
        if shared_length == 0:
            continue
        diffs = [
            abs(float(expert_jerk[index]) - float(learner_jerk[index]))
            for index in range(shared_length)
        ]
        channel_differences.append(clamp(safe_average(diffs), 0.0, 1.0))

    if not channel_differences:
        return 1.0

    mean_jerk_diff = safe_average(channel_differences)
    return round_metric(clamp(1.0 - mean_jerk_diff, 0.0, 1.0))


def _jerk_proxy_series(speed_series: list[Number]) -> list[float]:
    if len(speed_series) < 3:
        return []
    accelerations = [
        abs(float(speed_series[index]) - float(speed_series[index - 1]))
        for index in range(1, len(speed_series))
    ]
    if len(accelerations) < 2:
        return []
    return [
        abs(accelerations[index] - accelerations[index - 1])
        for index in range(1, len(accelerations))
    ]
