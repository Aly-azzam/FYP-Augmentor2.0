from __future__ import annotations

import math
from typing import Any

import numpy as np


def compute_region_metrics(raw_payload: dict[str, Any]) -> dict[str, Any]:
    frames = raw_payload.get("frames", [])
    regions = [region for region in raw_payload.get("regions", []) if region.get("valid")]
    mask_areas = np.array(
        [float(frame.get("mask_area", 0.0)) for frame in frames if frame.get("mask_area", 0) > 0],
        dtype=np.float32,
    )

    if regions:
        radii = np.array([float(region["r"]) for region in regions], dtype=np.float32)
        region_areas = math.pi * np.square(radii)
        region_points = np.array([[region["cx"], region["cy"]] for region in regions], dtype=np.float32)
    else:
        region_areas = np.array([], dtype=np.float32)
        region_points = np.empty((0, 2), dtype=np.float32)

    region_steps = (
        np.linalg.norm(np.diff(region_points, axis=0), axis=1)
        if len(region_points) >= 2
        else np.array([], dtype=np.float32)
    )
    jump_threshold = max(float(np.median(region_steps) * 3.0), 50.0) if len(region_steps) else 50.0
    region_jump_count = int(sum(float(step) > jump_threshold for step in region_steps))

    mask_area_variation = _coefficient_of_variation(mask_areas)
    region_step_std = float(region_steps.std()) if len(region_steps) else 0.0
    mask_score = max(0.0, min(1.0, 1.0 - mask_area_variation))
    jump_penalty = min(0.4, region_jump_count * 0.1)
    movement_score = max(0.0, min(1.0, 1.0 - (region_step_std / 45.0)))
    region_stability_score = max(0.0, min(1.0, (mask_score * 0.6 + movement_score * 0.4) - jump_penalty))

    return {
        "average_region_area": round(float(region_areas.mean()), 3) if len(region_areas) else 0.0,
        "average_mask_area": round(float(mask_areas.mean()), 3) if len(mask_areas) else 0.0,
        "region_stability_score": round(region_stability_score, 4),
        "mask_area_variation": round(mask_area_variation, 4),
        "region_jump_count": region_jump_count,
        "region_jump_threshold": round(jump_threshold, 3),
        "region_step_distance_std": round(region_step_std, 3),
    }


def _coefficient_of_variation(values: np.ndarray) -> float:
    if len(values) == 0:
        return 0.0
    mean_value = float(values.mean())
    if mean_value <= 1e-6:
        return 0.0
    return float(values.std() / mean_value)
