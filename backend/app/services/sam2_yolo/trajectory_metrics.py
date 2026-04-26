from __future__ import annotations

from typing import Any

import numpy as np


def compute_trajectory_metrics(raw_payload: dict[str, Any]) -> dict[str, Any]:
    frames = raw_payload.get("frames", [])
    trajectory_points = [
        point
        for point in raw_payload.get("trajectory", {}).get("points", [])
        if point.get("valid") and point.get("x") is not None and point.get("y") is not None
    ]
    lost_tracking_count = sum(1 for frame in frames if not frame.get("tracking_valid"))

    if not trajectory_points:
        return {
            "average_perpendicular_distance": None,
            "max_left_deviation": None,
            "max_right_deviation": None,
            "horizontal_drift_range": None,
            "vertical_drift_range": None,
            "smoothness_score": 0.0,
            "trajectory_stability_score": 0.0,
            "jump_count": 0,
            "lost_tracking_count": lost_tracking_count,
            "fitted_line": None,
        }

    points = np.array([[point["x"], point["y"]] for point in trajectory_points], dtype=np.float32)
    xs = points[:, 0]
    ys = points[:, 1]
    horizontal_drift = float(xs.max() - xs.min())
    vertical_drift = float(ys.max() - ys.min())

    if len(points) == 1:
        return {
            "average_perpendicular_distance": 0.0,
            "max_left_deviation": 0.0,
            "max_right_deviation": 0.0,
            "horizontal_drift_range": round(horizontal_drift, 3),
            "vertical_drift_range": round(vertical_drift, 3),
            "smoothness_score": 1.0,
            "trajectory_stability_score": 0.5 if lost_tracking_count == 0 else 0.25,
            "jump_count": 0,
            "lost_tracking_count": lost_tracking_count,
            "fitted_line": None,
        }

    mean_point = points.mean(axis=0)
    _, _, vh = np.linalg.svd(points - mean_point, full_matrices=False)
    direction = vh[0]
    direction = direction / max(float(np.linalg.norm(direction)), 1e-6)
    normal = np.array([-direction[1], direction[0]], dtype=np.float32)

    centered = points - mean_point
    signed_distances = centered @ normal
    perpendicular_distances = np.abs(signed_distances)
    projections = centered @ direction
    line_start = mean_point + direction * projections.min()
    line_end = mean_point + direction * projections.max()

    steps = np.linalg.norm(np.diff(points, axis=0), axis=1)
    mean_step = float(steps.mean()) if len(steps) else 0.0
    step_std = float(steps.std()) if len(steps) else 0.0
    jump_threshold = max(float(np.median(steps) * 3.0), 50.0) if len(steps) else 50.0
    jump_count = int(sum(float(step) > jump_threshold for step in steps))

    avg_perpendicular = float(perpendicular_distances.mean())
    smoothness_score = _score_from_error(step_std, scale=35.0)
    line_score = _score_from_error(avg_perpendicular, scale=45.0)
    loss_penalty = min(0.5, lost_tracking_count / max(len(frames), 1))
    jump_penalty = min(0.4, jump_count * 0.1)
    stability_score = max(0.0, min(1.0, (line_score * 0.7 + smoothness_score * 0.3) - loss_penalty - jump_penalty))

    return {
        "average_perpendicular_distance": round(avg_perpendicular, 3),
        "max_left_deviation": round(float(min(0.0, signed_distances.min())), 3),
        "max_right_deviation": round(float(max(0.0, signed_distances.max())), 3),
        "horizontal_drift_range": round(horizontal_drift, 3),
        "vertical_drift_range": round(vertical_drift, 3),
        "smoothness_score": round(smoothness_score, 4),
        "trajectory_stability_score": round(stability_score, 4),
        "jump_count": jump_count,
        "lost_tracking_count": lost_tracking_count,
        "jump_threshold": round(jump_threshold, 3),
        "mean_step_distance": round(mean_step, 3),
        "step_distance_std": round(step_std, 3),
        "fitted_line": {
            "point": [round(float(value), 3) for value in mean_point.tolist()],
            "direction": [round(float(value), 6) for value in direction.tolist()],
            "start_point": [round(float(value), 3) for value in line_start.tolist()],
            "end_point": [round(float(value), 3) for value in line_end.tolist()],
        },
    }


def _score_from_error(value: float, *, scale: float) -> float:
    return max(0.0, min(1.0, 1.0 - (float(value) / max(scale, 1e-6))))
