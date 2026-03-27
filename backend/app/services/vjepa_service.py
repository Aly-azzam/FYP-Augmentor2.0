"""Deterministic V-JEPA-inspired semantic phase analysis from motion dynamics."""

from __future__ import annotations

import math
from typing import Any

from app.schemas.motion_schema import MotionRepresentationOutput


def analyze_video_semantics(motion_data: MotionRepresentationOutput | dict[str, Any]) -> dict[str, Any]:
    """Analyze motion dynamics and return semantic action phases."""
    frames = _extract_frames(motion_data)
    if not frames:
        return {
            "phases": [
                {
                    "label": "preparation phase",
                    "start_frame": 0,
                    "end_frame": 0,
                }
            ],
            "motion_intensity_series": [],
        }

    frame_indices = [_frame_index(frame, idx) for idx, frame in enumerate(frames)]
    intensities = [_motion_intensity(frame) for frame in frames]
    labels = _classify_phase_labels(intensities)
    phases = _merge_phase_segments(labels, frame_indices)

    return {
        "phases": phases,
        "motion_intensity_series": [round(value, 4) for value in intensities],
    }


def _extract_frames(motion_data: MotionRepresentationOutput | dict[str, Any]) -> list[Any]:
    if isinstance(motion_data, dict):
        frames = motion_data.get("frames", [])
    else:
        frames = getattr(motion_data, "frames", [])
    if isinstance(frames, list):
        return frames
    return []


def _frame_index(frame: Any, fallback: int) -> int:
    if isinstance(frame, dict):
        return int(frame.get("frame_index", fallback))
    return int(getattr(frame, "frame_index", fallback))


def _motion_intensity(frame: Any) -> float:
    left = _get_hand_features(frame, "left_hand_features")
    right = _get_hand_features(frame, "right_hand_features")

    channels = [
        _vector_magnitude(_get_list(left, "wrist_velocity")),
        _vector_magnitude(_get_list(right, "wrist_velocity")),
        _vector_magnitude(_get_list(left, "palm_velocity")),
        _vector_magnitude(_get_list(right, "palm_velocity")),
    ]
    return sum(channels) / float(len(channels))


def _get_hand_features(frame: Any, key: str) -> Any:
    if isinstance(frame, dict):
        return frame.get(key, {})
    return getattr(frame, key, {})


def _get_list(hand_features: Any, key: str) -> list[float]:
    if isinstance(hand_features, dict):
        value = hand_features.get(key, [])
    else:
        value = getattr(hand_features, key, [])
    if not isinstance(value, list):
        return [0.0, 0.0, 0.0]
    padded = [float(v) for v in value[:3]]
    while len(padded) < 3:
        padded.append(0.0)
    return padded


def _vector_magnitude(vector: list[float]) -> float:
    return math.sqrt(sum(component * component for component in vector))


def _classify_phase_labels(intensities: list[float]) -> list[str]:
    if not intensities:
        return []
    mean_intensity = sum(intensities) / float(len(intensities))
    variance = sum((value - mean_intensity) ** 2 for value in intensities) / float(len(intensities))
    std_intensity = math.sqrt(max(variance, 0.0))

    low_threshold = max(0.0, mean_intensity - (0.25 * std_intensity))
    high_threshold = mean_intensity + (0.25 * std_intensity)

    diffs = [0.0]
    for index in range(1, len(intensities)):
        diffs.append(float(intensities[index] - intensities[index - 1]))
    diff_variance = sum(diff * diff for diff in diffs) / float(len(diffs))
    slope_threshold = max(0.01, 0.25 * math.sqrt(diff_variance))

    labels: list[str] = []
    for index, intensity in enumerate(intensities):
        slope = diffs[index]
        if intensity <= low_threshold:
            labels.append("preparation phase")
        elif slope <= -slope_threshold:
            labels.append("finishing phase")
        elif slope >= slope_threshold or intensity >= high_threshold:
            labels.append("execution phase")
        else:
            labels.append("steady phase")

    return labels


def _merge_phase_segments(labels: list[str], frame_indices: list[int]) -> list[dict[str, Any]]:
    if not labels:
        return [{"label": "preparation phase", "start_frame": 0, "end_frame": 0}]

    segments: list[dict[str, Any]] = []
    segment_label = labels[0]
    start_index = 0

    for index in range(1, len(labels)):
        if labels[index] != segment_label:
            segments.append(
                {
                    "label": segment_label,
                    "start_frame": int(frame_indices[start_index]),
                    "end_frame": int(frame_indices[index - 1]),
                }
            )
            segment_label = labels[index]
            start_index = index

    segments.append(
        {
            "label": segment_label,
            "start_frame": int(frame_indices[start_index]),
            "end_frame": int(frame_indices[-1]),
        }
    )
    return segments
