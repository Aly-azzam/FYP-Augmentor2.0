"""Helpers for loading and validating motion JSON data.

This module is part of the Evaluation Engine. Its job is to:
- load expert and learner motion data from JSON files or dictionaries
- validate the expected top-level structure
- prepare a clean paired payload for later comparison logic
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any


REQUIRED_TOP_LEVEL_FIELDS = (
    "video_id",
    "angle_series",
    "joint_trajectories",
    "velocity_profiles",
    "tool_motion",
)


class MotionDataError(Exception):
    """Base exception for motion data problems."""


class MissingFieldError(MotionDataError):
    """Raised when a required top-level field is missing."""


class InvalidStructureError(MotionDataError):
    """Raised when motion data has an invalid structure."""


def load_motion_data(source: str | Path | dict[str, Any]) -> dict[str, Any]:
    """Load motion data from a file path or an already-built dictionary."""
    if isinstance(source, dict):
        return source

    path = Path(source)
    if not path.exists():
        raise FileNotFoundError(f"Motion data file not found: {path}")

    try:
        with path.open("r", encoding="utf-8") as file:
            data = json.load(file)
    except json.JSONDecodeError as exc:
        raise InvalidStructureError(f"Invalid JSON in motion data file: {path}") from exc

    if not isinstance(data, dict):
        raise InvalidStructureError("Motion data must be a JSON object at the top level.")

    return data


def validate_motion_data(data: dict[str, Any]) -> None:
    """Validate the required fields and basic structure of motion data."""
    if not isinstance(data, dict):
        raise InvalidStructureError("Motion data must be a dictionary.")

    for field_name in REQUIRED_TOP_LEVEL_FIELDS:
        if field_name not in data:
            raise MissingFieldError(f"Missing required field: '{field_name}'")

    if not isinstance(data["video_id"], str) or not data["video_id"].strip():
        raise InvalidStructureError("'video_id' must be a non-empty string.")

    _validate_mapping_field(data, "angle_series")
    _validate_mapping_field(data, "joint_trajectories")
    _validate_mapping_field(data, "velocity_profiles")
    _validate_mapping_field(data, "tool_motion")

    _validate_number_series_mapping(data["angle_series"], "angle_series")
    _validate_point_series_mapping(data["joint_trajectories"], "joint_trajectories")
    _validate_number_series_mapping(data["velocity_profiles"], "velocity_profiles")
    _validate_point_series_mapping(data["tool_motion"], "tool_motion")


def pair_motion_data(
    expert_data: dict[str, Any],
    learner_data: dict[str, Any],
) -> dict[str, Any]:
    """Validate and pair expert and learner motion data for later comparison."""
    validate_motion_data(expert_data)
    validate_motion_data(learner_data)

    return {
        "expert_video_id": expert_data["video_id"],
        "learner_video_id": learner_data["video_id"],
        "expert_motion": expert_data,
        "learner_motion": learner_data,
        "shared_keys": _build_shared_keys(expert_data, learner_data),
    }


def _validate_mapping_field(data: dict[str, Any], field_name: str) -> None:
    """Ensure a required field is a dictionary-like mapping."""
    value = data[field_name]
    if not isinstance(value, dict):
        raise InvalidStructureError(f"'{field_name}' must be a dictionary.")


def _validate_number_series_mapping(mapping: dict[str, Any], field_name: str) -> None:
    """Ensure a mapping contains only lists of numeric values."""
    for key, value in mapping.items():
        if not isinstance(value, list):
            raise InvalidStructureError(f"'{field_name}.{key}' must be a list.")

        for item in value:
            if not isinstance(item, (int, float)):
                raise InvalidStructureError(
                    f"'{field_name}.{key}' must contain only numeric values."
                )


def _validate_point_series_mapping(mapping: dict[str, Any], field_name: str) -> None:
    """Ensure a mapping contains only lists of 2D numeric points."""
    for key, value in mapping.items():
        if not isinstance(value, list):
            raise InvalidStructureError(f"'{field_name}.{key}' must be a list.")

        for point in value:
            if not isinstance(point, list) or len(point) < 2:
                raise InvalidStructureError(
                    f"'{field_name}.{key}' must contain 2D points like [x, y]."
                )

            if not isinstance(point[0], (int, float)) or not isinstance(point[1], (int, float)):
                raise InvalidStructureError(
                    f"'{field_name}.{key}' must contain only numeric 2D points."
                )


def pair_motion_data_with_alignment(
    expert_data: dict[str, Any],
    learner_data: dict[str, Any],
    aligned_pairs: list[dict[str, int]],
) -> dict[str, Any]:
    """Build metric-compatible paired motion data using aligned frame pairs."""
    if not isinstance(aligned_pairs, list) or not aligned_pairs:
        raise InvalidStructureError("'aligned_pairs' must be a non-empty list.")

    expert_frames = expert_data.get("frames")
    learner_frames = learner_data.get("frames")
    if not isinstance(expert_frames, list) or not isinstance(learner_frames, list):
        raise InvalidStructureError("Aligned pairing requires 'frames' arrays for both inputs.")

    expert_aligned_frames: list[dict[str, Any]] = []
    learner_aligned_frames: list[dict[str, Any]] = []
    for pair in aligned_pairs:
        expert_index = pair.get("expert_index")
        learner_index = pair.get("learner_index")
        if not isinstance(expert_index, int) or not isinstance(learner_index, int):
            raise InvalidStructureError("Each aligned pair must contain integer indices.")
        if expert_index < 0 or learner_index < 0:
            raise InvalidStructureError("Aligned pair indices must be >= 0.")
        if expert_index >= len(expert_frames) or learner_index >= len(learner_frames):
            continue
        expert_aligned_frames.append(expert_frames[expert_index])
        learner_aligned_frames.append(learner_frames[learner_index])

    expert_aligned = _frames_to_metric_motion(
        video_id=str(expert_data.get("video_id", "expert")),
        aligned_frames=expert_aligned_frames,
    )
    learner_aligned = _frames_to_metric_motion(
        video_id=str(learner_data.get("video_id", "learner")),
        aligned_frames=learner_aligned_frames,
    )
    modern_aligned_channels = _build_modern_aligned_channels(
        expert_aligned_frames=expert_aligned_frames,
        learner_aligned_frames=learner_aligned_frames,
        alignment_path=aligned_pairs,
    )

    return {
        "expert_video_id": expert_aligned["video_id"],
        "learner_video_id": learner_aligned["video_id"],
        "expert_motion": expert_aligned,
        "learner_motion": learner_aligned,
        "shared_keys": _build_shared_keys(expert_aligned, learner_aligned),
        "modern_aligned_channels": modern_aligned_channels,
    }


def _build_shared_keys(expert_data: dict[str, Any], learner_data: dict[str, Any]) -> dict[str, list[str]]:
    return {
        "angle_series": sorted(
            set(expert_data["angle_series"].keys()) & set(learner_data["angle_series"].keys())
        ),
        "joint_trajectories": sorted(
            set(expert_data["joint_trajectories"].keys())
            & set(learner_data["joint_trajectories"].keys())
        ),
        "velocity_profiles": sorted(
            set(expert_data["velocity_profiles"].keys())
            & set(learner_data["velocity_profiles"].keys())
        ),
        "tool_motion": sorted(
            set(expert_data["tool_motion"].keys()) & set(learner_data["tool_motion"].keys())
        ),
    }


def _frames_to_metric_motion(video_id: str, aligned_frames: list[dict[str, Any]]) -> dict[str, Any]:
    angle_series: dict[str, list[float]] = {}
    joint_trajectories: dict[str, list[list[float]]] = {}
    velocity_profiles: dict[str, list[float]] = {}
    tool_motion: dict[str, list[list[float]]] = {}

    for frame in aligned_frames:
        left_hand = _extract_hand_features(frame, "left_hand_features")
        right_hand = _extract_hand_features(frame, "right_hand_features")

        _append_angle_series(angle_series, "left", left_hand)
        _append_angle_series(angle_series, "right", right_hand)

        _append_joint_trajectory(joint_trajectories, "left", left_hand)
        _append_joint_trajectory(joint_trajectories, "right", right_hand)

        _append_velocity_profile(velocity_profiles, "left", left_hand)
        _append_velocity_profile(velocity_profiles, "right", right_hand)

    return {
        "video_id": video_id,
        "angle_series": angle_series,
        "joint_trajectories": joint_trajectories,
        "velocity_profiles": velocity_profiles,
        "tool_motion": tool_motion,
    }


def _extract_hand_features(frame: dict[str, Any], hand_key: str) -> dict[str, Any]:
    if not isinstance(frame, dict):
        raise InvalidStructureError("Aligned frame must be a dictionary.")
    hand_features = frame.get(hand_key)
    if not isinstance(hand_features, dict):
        raise InvalidStructureError(f"Aligned frame must include '{hand_key}'.")
    return hand_features


def _append_angle_series(
    angle_series: dict[str, list[float]],
    hand_label: str,
    hand_features: dict[str, Any],
) -> None:
    joint_angles = hand_features.get("joint_angles", {})
    if not isinstance(joint_angles, dict):
        return
    for angle_name, angle_value in joint_angles.items():
        key = f"{hand_label}_{angle_name}"
        angle_series.setdefault(key, []).append(float(angle_value))


def _append_joint_trajectory(
    joint_trajectories: dict[str, list[list[float]]],
    hand_label: str,
    hand_features: dict[str, Any],
) -> None:
    wrist = _extract_xy(hand_features.get("wrist_position"))
    palm = _extract_xy(hand_features.get("palm_center"))
    fingertip_positions = hand_features.get("fingertip_positions", {})
    index_tip = _extract_xy(
        fingertip_positions.get("index_tip") if isinstance(fingertip_positions, dict) else None
    )

    joint_trajectories.setdefault(f"{hand_label}_wrist", []).append(wrist)
    joint_trajectories.setdefault(f"{hand_label}_palm_center", []).append(palm)
    joint_trajectories.setdefault(f"{hand_label}_index_tip", []).append(index_tip)


def _append_velocity_profile(
    velocity_profiles: dict[str, list[float]],
    hand_label: str,
    hand_features: dict[str, Any],
) -> None:
    wrist_velocity = _extract_xyz(hand_features.get("wrist_velocity"))
    palm_velocity = _extract_xyz(hand_features.get("palm_velocity"))
    velocity_profiles.setdefault(f"{hand_label}_wrist_velocity_mag", []).append(
        _vector_magnitude(wrist_velocity)
    )
    velocity_profiles.setdefault(f"{hand_label}_palm_velocity_mag", []).append(
        _vector_magnitude(palm_velocity)
    )


def _extract_xy(value: Any) -> list[float]:
    if isinstance(value, list) and len(value) >= 2:
        return [float(value[0]), float(value[1])]
    return [0.0, 0.0]


def _extract_xyz(value: Any) -> list[float]:
    if isinstance(value, list):
        padded = [float(v) for v in value[:3]]
        while len(padded) < 3:
            padded.append(0.0)
        return padded
    return [0.0, 0.0, 0.0]


def _vector_magnitude(vector: list[float]) -> float:
    return math.sqrt(sum(component * component for component in vector))


def _build_modern_aligned_channels(
    *,
    expert_aligned_frames: list[dict[str, Any]],
    learner_aligned_frames: list[dict[str, Any]],
    alignment_path: list[dict[str, int]],
) -> dict[str, Any]:
    trajectory_channels: dict[str, dict[str, list[list[float]]]] = {
        key: {"expert": [], "learner": []}
        for key in (
            "left_wrist",
            "left_palm_center",
            "left_index_tip",
            "right_wrist",
            "right_palm_center",
            "right_index_tip",
        )
    }
    velocity_channels: dict[str, dict[str, list[float]]] = {
        key: {"expert": [], "learner": []}
        for key in (
            "left_wrist_velocity_mag",
            "left_palm_velocity_mag",
            "right_wrist_velocity_mag",
            "right_palm_velocity_mag",
        )
    }
    angle_channels: dict[str, dict[str, list[float]]] = {
        key: {"expert": [], "learner": []}
        for key in (
            "left_wrist_index_mcp_index_tip",
            "left_wrist_middle_mcp_middle_tip",
            "left_wrist_ring_mcp_ring_tip",
            "left_wrist_index_mcp_pinky_mcp",
            "right_wrist_index_mcp_index_tip",
            "right_wrist_middle_mcp_middle_tip",
            "right_wrist_ring_mcp_ring_tip",
            "right_wrist_index_mcp_pinky_mcp",
        )
    }
    hand_shape_channels: dict[str, dict[str, list[float]]] = {
        key: {"expert": [], "learner": []}
        for key in (
            "left_hand_openness",
            "left_pinch_distance",
            "left_finger_spread",
            "right_hand_openness",
            "right_pinch_distance",
            "right_finger_spread",
        )
    }
    expert_timestamps: list[float] = []
    learner_timestamps: list[float] = []

    for expert_frame, learner_frame in zip(expert_aligned_frames, learner_aligned_frames):
        expert_timestamps.append(float(expert_frame.get("timestamp_sec", 0.0)))
        learner_timestamps.append(float(learner_frame.get("timestamp_sec", 0.0)))

        _append_hand_channels(
            channels=trajectory_channels,
            hand_prefix="left",
            expert_hand=_extract_hand_features(expert_frame, "left_hand_features"),
            learner_hand=_extract_hand_features(learner_frame, "left_hand_features"),
            value_kind="trajectory",
        )
        _append_hand_channels(
            channels=trajectory_channels,
            hand_prefix="right",
            expert_hand=_extract_hand_features(expert_frame, "right_hand_features"),
            learner_hand=_extract_hand_features(learner_frame, "right_hand_features"),
            value_kind="trajectory",
        )
        _append_hand_channels(
            channels=velocity_channels,
            hand_prefix="left",
            expert_hand=_extract_hand_features(expert_frame, "left_hand_features"),
            learner_hand=_extract_hand_features(learner_frame, "left_hand_features"),
            value_kind="velocity",
        )
        _append_hand_channels(
            channels=velocity_channels,
            hand_prefix="right",
            expert_hand=_extract_hand_features(expert_frame, "right_hand_features"),
            learner_hand=_extract_hand_features(learner_frame, "right_hand_features"),
            value_kind="velocity",
        )
        _append_hand_channels(
            channels=angle_channels,
            hand_prefix="left",
            expert_hand=_extract_hand_features(expert_frame, "left_hand_features"),
            learner_hand=_extract_hand_features(learner_frame, "left_hand_features"),
            value_kind="angles",
        )
        _append_hand_channels(
            channels=angle_channels,
            hand_prefix="right",
            expert_hand=_extract_hand_features(expert_frame, "right_hand_features"),
            learner_hand=_extract_hand_features(learner_frame, "right_hand_features"),
            value_kind="angles",
        )
        _append_hand_channels(
            channels=hand_shape_channels,
            hand_prefix="left",
            expert_hand=_extract_hand_features(expert_frame, "left_hand_features"),
            learner_hand=_extract_hand_features(learner_frame, "left_hand_features"),
            value_kind="hand_shape",
        )
        _append_hand_channels(
            channels=hand_shape_channels,
            hand_prefix="right",
            expert_hand=_extract_hand_features(expert_frame, "right_hand_features"),
            learner_hand=_extract_hand_features(learner_frame, "right_hand_features"),
            value_kind="hand_shape",
        )

    return {
        "alignment_path": alignment_path,
        "trajectory_channels": trajectory_channels,
        "velocity_channels": velocity_channels,
        "angle_channels": angle_channels,
        "hand_shape_channels": hand_shape_channels,
        "expert_timestamps": expert_timestamps,
        "learner_timestamps": learner_timestamps,
    }


def _append_hand_channels(
    *,
    channels: dict[str, dict[str, list[Any]]],
    hand_prefix: str,
    expert_hand: dict[str, Any],
    learner_hand: dict[str, Any],
    value_kind: str,
) -> None:
    if value_kind == "trajectory":
        expert_points = _trajectory_points_for_hand(expert_hand)
        learner_points = _trajectory_points_for_hand(learner_hand)
        channels[f"{hand_prefix}_wrist"]["expert"].append(expert_points["wrist"])
        channels[f"{hand_prefix}_wrist"]["learner"].append(learner_points["wrist"])
        channels[f"{hand_prefix}_palm_center"]["expert"].append(expert_points["palm_center"])
        channels[f"{hand_prefix}_palm_center"]["learner"].append(learner_points["palm_center"])
        channels[f"{hand_prefix}_index_tip"]["expert"].append(expert_points["index_tip"])
        channels[f"{hand_prefix}_index_tip"]["learner"].append(learner_points["index_tip"])
        return

    if value_kind == "velocity":
        expert_wrist_vel = _vector_magnitude(_extract_xyz(expert_hand.get("wrist_velocity")))
        learner_wrist_vel = _vector_magnitude(_extract_xyz(learner_hand.get("wrist_velocity")))
        expert_palm_vel = _vector_magnitude(_extract_xyz(expert_hand.get("palm_velocity")))
        learner_palm_vel = _vector_magnitude(_extract_xyz(learner_hand.get("palm_velocity")))
        channels[f"{hand_prefix}_wrist_velocity_mag"]["expert"].append(expert_wrist_vel)
        channels[f"{hand_prefix}_wrist_velocity_mag"]["learner"].append(learner_wrist_vel)
        channels[f"{hand_prefix}_palm_velocity_mag"]["expert"].append(expert_palm_vel)
        channels[f"{hand_prefix}_palm_velocity_mag"]["learner"].append(learner_palm_vel)
        return

    if value_kind == "angles":
        joint_names = (
            "wrist_index_mcp_index_tip",
            "wrist_middle_mcp_middle_tip",
            "wrist_ring_mcp_ring_tip",
            "wrist_index_mcp_pinky_mcp",
        )
        expert_joint_angles = expert_hand.get("joint_angles", {})
        learner_joint_angles = learner_hand.get("joint_angles", {})
        for joint_name in joint_names:
            channels[f"{hand_prefix}_{joint_name}"]["expert"].append(
                float(expert_joint_angles.get(joint_name, 0.0))
            )
            channels[f"{hand_prefix}_{joint_name}"]["learner"].append(
                float(learner_joint_angles.get(joint_name, 0.0))
            )
        return

    if value_kind == "hand_shape":
        channels[f"{hand_prefix}_hand_openness"]["expert"].append(float(expert_hand.get("hand_openness", 0.0)))
        channels[f"{hand_prefix}_hand_openness"]["learner"].append(float(learner_hand.get("hand_openness", 0.0)))
        channels[f"{hand_prefix}_pinch_distance"]["expert"].append(float(expert_hand.get("pinch_distance", 0.0)))
        channels[f"{hand_prefix}_pinch_distance"]["learner"].append(float(learner_hand.get("pinch_distance", 0.0)))
        channels[f"{hand_prefix}_finger_spread"]["expert"].append(float(expert_hand.get("finger_spread", 0.0)))
        channels[f"{hand_prefix}_finger_spread"]["learner"].append(float(learner_hand.get("finger_spread", 0.0)))
        return

    raise InvalidStructureError(f"Unknown channel value_kind: {value_kind}")


def _trajectory_points_for_hand(hand_features: dict[str, Any]) -> dict[str, list[float]]:
    fingertip_positions = hand_features.get("fingertip_positions", {})
    index_tip_raw = fingertip_positions.get("index_tip") if isinstance(fingertip_positions, dict) else None
    return {
        "wrist": _extract_xy(hand_features.get("wrist_position")),
        "palm_center": _extract_xy(hand_features.get("palm_center")),
        "index_tip": _extract_xy(index_tip_raw),
    }


def build_angle_error_series(modern_aligned_channels: dict[str, Any]) -> list[float]:
    """Build per-aligned-frame normalized angle error [0,1]."""
    angle_channels = modern_aligned_channels.get("angle_channels", {})
    if not isinstance(angle_channels, dict) or not angle_channels:
        return []
    series_length = _infer_channel_length(angle_channels)
    errors: list[float] = []
    for index in range(series_length):
        channel_diffs: list[float] = []
        for payload in angle_channels.values():
            if not isinstance(payload, dict):
                continue
            expert_values = payload.get("expert", [])
            learner_values = payload.get("learner", [])
            if index >= len(expert_values) or index >= len(learner_values):
                continue
            diff = abs(float(expert_values[index]) - float(learner_values[index])) / 180.0
            channel_diffs.append(_clamp01(diff))
        errors.append(_safe_mean(channel_diffs))
    return errors


def build_speed_error_series(modern_aligned_channels: dict[str, Any]) -> list[float]:
    """Build per-aligned-frame normalized speed error [0,1]."""
    velocity_channels = modern_aligned_channels.get("velocity_channels", {})
    if not isinstance(velocity_channels, dict) or not velocity_channels:
        return []
    series_length = _infer_channel_length(velocity_channels)
    errors: list[float] = []
    for index in range(series_length):
        channel_diffs: list[float] = []
        for payload in velocity_channels.values():
            if not isinstance(payload, dict):
                continue
            expert_values = payload.get("expert", [])
            learner_values = payload.get("learner", [])
            if index >= len(expert_values) or index >= len(learner_values):
                continue
            diff = abs(float(expert_values[index]) - float(learner_values[index]))
            channel_diffs.append(_clamp01(diff))
        errors.append(_safe_mean(channel_diffs))
    return errors


def build_trajectory_error_series(modern_aligned_channels: dict[str, Any]) -> list[float]:
    """Build per-aligned-frame normalized trajectory error [0,1]."""
    trajectory_channels = modern_aligned_channels.get("trajectory_channels", {})
    if not isinstance(trajectory_channels, dict) or not trajectory_channels:
        return []
    series_length = _infer_channel_length(trajectory_channels)
    errors: list[float] = []
    for index in range(series_length):
        channel_diffs: list[float] = []
        for payload in trajectory_channels.values():
            if not isinstance(payload, dict):
                continue
            expert_points = payload.get("expert", [])
            learner_points = payload.get("learner", [])
            if index >= len(expert_points) or index >= len(learner_points):
                continue
            expert_point = expert_points[index]
            learner_point = learner_points[index]
            if not isinstance(expert_point, list) or not isinstance(learner_point, list):
                continue
            if len(expert_point) < 2 or len(learner_point) < 2:
                continue
            dx = float(expert_point[0]) - float(learner_point[0])
            dy = float(expert_point[1]) - float(learner_point[1])
            dist = math.sqrt((dx * dx) + (dy * dy)) / math.sqrt(2.0)
            channel_diffs.append(_clamp01(dist))
        errors.append(_safe_mean(channel_diffs))
    return errors


def detect_top_error_segment(
    *,
    error_series: list[float],
    alignment_path: list[dict[str, int]],
    error_type: str,
    label: str,
) -> dict[str, Any]:
    """Detect the highest-severity contiguous segment for one error series."""
    if not error_series:
        return {
            "error_type": error_type,
            "label": label,
            "start_aligned_index": 0,
            "end_aligned_index": 0,
            "start_expert_frame": 0,
            "end_expert_frame": 0,
            "start_learner_frame": 0,
            "end_learner_frame": 0,
            "severity": 0.0,
        }

    smoothed = _moving_average(error_series, window=3)
    mean_value = _safe_mean(smoothed)
    std_value = _safe_std(smoothed, mean_value)
    threshold = max(mean_value + (0.5 * std_value), 0.1)

    segments: list[tuple[int, int]] = []
    segment_start: int | None = None
    for index, value in enumerate(smoothed):
        if value >= threshold:
            if segment_start is None:
                segment_start = index
        elif segment_start is not None:
            segments.append((segment_start, index - 1))
            segment_start = None
    if segment_start is not None:
        segments.append((segment_start, len(smoothed) - 1))

    if not segments:
        max_index = max(range(len(smoothed)), key=lambda idx: smoothed[idx])
        segments = [(max_index, max_index)]

    best_start, best_end = max(
        segments,
        key=lambda bounds: _safe_mean(smoothed[bounds[0] : bounds[1] + 1]),
    )
    severity = _clamp01(_safe_mean(smoothed[best_start : best_end + 1]))

    start_pair = alignment_path[min(best_start, len(alignment_path) - 1)] if alignment_path else {}
    end_pair = alignment_path[min(best_end, len(alignment_path) - 1)] if alignment_path else {}

    return {
        "error_type": error_type,
        "label": label,
        "start_aligned_index": int(best_start),
        "end_aligned_index": int(best_end),
        "start_expert_frame": int(start_pair.get("expert_index", 0)),
        "end_expert_frame": int(end_pair.get("expert_index", 0)),
        "start_learner_frame": int(start_pair.get("learner_index", 0)),
        "end_learner_frame": int(end_pair.get("learner_index", 0)),
        "severity": float(round(severity, 4)),
    }


def detect_key_error_moments(modern_aligned_channels: dict[str, Any]) -> list[dict[str, Any]]:
    """Return top localized error moments for angle/speed/trajectory."""
    alignment_path = modern_aligned_channels.get("alignment_path", [])
    if not isinstance(alignment_path, list):
        alignment_path = []

    angle_series = build_angle_error_series(modern_aligned_channels)
    speed_series = build_speed_error_series(modern_aligned_channels)
    trajectory_series = build_trajectory_error_series(modern_aligned_channels)

    return [
        detect_top_error_segment(
            error_series=angle_series,
            alignment_path=alignment_path,
            error_type="angle_error",
            label="wrist angle issue",
        ),
        detect_top_error_segment(
            error_series=speed_series,
            alignment_path=alignment_path,
            error_type="speed_error",
            label="inconsistent speed",
        ),
        detect_top_error_segment(
            error_series=trajectory_series,
            alignment_path=alignment_path,
            error_type="trajectory_error",
            label="path drift",
        ),
    ]


def map_errors_to_phases(
    error_moments: list[dict[str, Any]],
    semantic_phases: dict[str, Any],
) -> list[dict[str, Any]]:
    """Attach best-overlap expert semantic phase info to each error moment."""
    if not isinstance(error_moments, list) or not error_moments:
        return []

    expert_semantics = semantic_phases.get("expert", {}) if isinstance(semantic_phases, dict) else {}
    phase_objects = expert_semantics.get("phases", []) if isinstance(expert_semantics, dict) else []
    if not isinstance(phase_objects, list):
        phase_objects = []

    normalized_phases: list[dict[str, Any]] = []
    for phase in phase_objects:
        phase_dict = _to_mapping(phase)
        start_frame = int(phase_dict.get("start_frame", 0))
        end_frame = int(phase_dict.get("end_frame", start_frame))
        if end_frame < start_frame:
            start_frame, end_frame = end_frame, start_frame
        normalized_phases.append(
            {
                "label": str(phase_dict.get("label", "unknown phase")),
                "start_frame": start_frame,
                "end_frame": end_frame,
            }
        )

    fused: list[dict[str, Any]] = []
    for error in error_moments:
        error_dict = _to_mapping(error)
        start_frame = int(error_dict.get("start_expert_frame", 0))
        end_frame = int(error_dict.get("end_expert_frame", start_frame))
        if end_frame < start_frame:
            start_frame, end_frame = end_frame, start_frame

        selected_phase = _select_phase_for_error(
            error_start=start_frame,
            error_end=end_frame,
            phases=normalized_phases,
        )
        if selected_phase is None:
            selected_phase = {
                "label": "unknown phase",
                "start_frame": start_frame,
                "end_frame": end_frame,
            }

        fused_error = dict(error_dict)
        fused_error["semantic_phase"] = selected_phase
        fused_error["semantic_label"] = f"{fused_error.get('label', 'error')} during {selected_phase['label']}"
        fused.append(fused_error)

    return fused


def _to_mapping(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if hasattr(value, "model_dump"):
        dumped = value.model_dump()
        if isinstance(dumped, dict):
            return dumped
    return {}


def _select_phase_for_error(
    *,
    error_start: int,
    error_end: int,
    phases: list[dict[str, Any]],
) -> dict[str, Any] | None:
    if not phases:
        return None

    best_overlap_phase: dict[str, Any] | None = None
    best_overlap = -1
    for phase in phases:
        overlap_start = max(error_start, int(phase["start_frame"]))
        overlap_end = min(error_end, int(phase["end_frame"]))
        overlap = max(0, overlap_end - overlap_start + 1)
        if overlap > best_overlap:
            best_overlap = overlap
            best_overlap_phase = phase

    if best_overlap_phase is not None and best_overlap > 0:
        return best_overlap_phase

    # Fallback: closest phase by boundary distance.
    def _distance(phase: dict[str, Any]) -> int:
        start = int(phase["start_frame"])
        end = int(phase["end_frame"])
        if error_end < start:
            return start - error_end
        if error_start > end:
            return error_start - end
        return 0

    return min(phases, key=_distance)


def _infer_channel_length(channels: dict[str, dict[str, list[Any]]]) -> int:
    for payload in channels.values():
        if not isinstance(payload, dict):
            continue
        expert_values = payload.get("expert")
        if isinstance(expert_values, list):
            return len(expert_values)
    return 0


def _safe_mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return sum(float(v) for v in values) / float(len(values))


def _safe_std(values: list[float], mean_value: float) -> float:
    if len(values) < 2:
        return 0.0
    variance = sum((float(v) - mean_value) ** 2 for v in values) / float(len(values))
    return math.sqrt(max(variance, 0.0))


def _moving_average(values: list[float], *, window: int) -> list[float]:
    if window <= 1 or len(values) <= 1:
        return [float(v) for v in values]
    radius = window // 2
    smoothed: list[float] = []
    for index in range(len(values)):
        start = max(0, index - radius)
        end = min(len(values), index + radius + 1)
        smoothed.append(_safe_mean([float(v) for v in values[start:end]]))
    return smoothed


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))
