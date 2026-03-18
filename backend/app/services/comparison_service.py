"""Helpers for loading and validating motion JSON data.

This module is part of the Evaluation Engine. Its job is to:
- load expert and learner motion data from JSON files or dictionaries
- validate the expected top-level structure
- prepare a clean paired payload for later comparison logic
"""

from __future__ import annotations

import json
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
        "shared_keys": {
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
        },
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
