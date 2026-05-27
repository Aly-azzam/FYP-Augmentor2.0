"""Merge trajectory, angle, and vibration errors into a single unified_errors.json.

Inputs
------
storage/evaluation/{run_id}/trajectory/trajectory_errors.json
storage/evaluation/{run_id}/angle/angle_errors.json
storage/evaluation/{run_id}/trajectory/aligned_corridor.json
vibration_errors (optional) — pre-built list from detect_vibration_errors()

Output
------
storage/evaluation/{run_id}/score/unified_errors.json
"""

from __future__ import annotations

import json
import os
from pathlib import Path


def compute_trajectory_bbox(error: dict, frame_checks: list[dict], padding: int = 250) -> dict:
    """Find all frame_checks whose frame_index falls between
    error.frame_start and error.frame_end, then compute bbox
    of their learner_x/learner_y with padding.
    """
    points_in_window = [
        fc for fc in frame_checks
        if error["frame_start"] <= fc["frame_index"] <= error["frame_end"]
        and fc.get("learner_x") is not None
        and fc.get("learner_y") is not None
    ]
    if not points_in_window:
        px = error["peak_location"]["x"]
        py = error["peak_location"]["y"]
        return {
            "x_min": round(px - 250, 2),
            "y_min": round(py - 250, 2),
            "x_max": round(px + 250, 2),
            "y_max": round(py + 250, 2),
        }

    xs = [fc["learner_x"] for fc in points_in_window]
    ys = [fc["learner_y"] for fc in points_in_window]

    return {
        "x_min": round(min(xs) - padding, 2),
        "y_min": round(min(ys) - padding, 2),
        "x_max": round(max(xs) + padding, 2),
        "y_max": round(max(ys) + padding, 2),
    }


def compute_angle_bbox(error: dict, frame_checks: list[dict], padding: int = 150) -> dict:
    """Bounding box around inside-corridor frames during the angle error window.

    These are the red-dot positions — scissors inside the corridor but holding
    the wrong angle.
    """
    points_in_window = [
        fc for fc in frame_checks
        if error["frame_start"] <= fc["frame_index"] <= error["frame_end"]
        and not fc.get("outside", True)
        and fc.get("learner_x") is not None
        and fc.get("learner_y") is not None
    ]
    if not points_in_window:
        px = error["peak_location"]["x"]
        py = error["peak_location"]["y"]
        return {
            "x_min": round(px - 150, 2),
            "y_min": round(py - 150, 2),
            "x_max": round(px + 150, 2),
            "y_max": round(py + 150, 2),
        }

    xs = [fc["learner_x"] for fc in points_in_window]
    ys = [fc["learner_y"] for fc in points_in_window]

    return {
        "x_min": round(min(xs) - padding, 2),
        "y_min": round(min(ys) - padding, 2),
        "x_max": round(max(xs) + padding, 2),
        "y_max": round(max(ys) + padding, 2),
    }


def merge_errors(
    trajectory_errors_path: str,
    angle_errors_path: str,
    aligned_corridor_path: str,
    output_dir: str,
    vibration_errors: list[dict] | None = None,
) -> dict:
    """Merge trajectory, angle, and vibration error events into one sorted unified list.

    Parameters
    ----------
    trajectory_errors_path:
        Path to ``trajectory_errors.json``.
    angle_errors_path:
        Path to ``angle_errors.json``.
    aligned_corridor_path:
        Path to ``aligned_corridor.json`` — used to compute bounding boxes for
        trajectory errors from actual learner positions.
    output_dir:
        Directory where ``unified_errors.json`` is written
        (typically ``storage/evaluation/{run_id}/score/``).
    vibration_errors:
        Optional list of pre-built error dicts from ``detect_vibration_errors()``.
        Each dict already contains a ``bounding_box`` and matches the unified
        schema.  Defaults to an empty list when not provided.

    Returns
    -------
    dict
        The full unified payload that was also written to disk.
    """
    with open(trajectory_errors_path) as fh:
        traj = json.load(fh)
    with open(angle_errors_path) as fh:
        angle = json.load(fh)

    corridor_available = Path(aligned_corridor_path).exists()
    frame_checks: list[dict] = []
    if corridor_available:
        with open(aligned_corridor_path) as f:
            corridor = json.load(f)
        frame_checks = corridor.get("frame_checks", [])

    all_errors: list[dict] = []

    for e in traj["error_events"]:
        bbox = compute_trajectory_bbox(e, frame_checks, padding=80) if corridor_available else None
        all_errors.append({
            "error_id": None,
            "error_type": "trajectory",
            "frame_start": e["frame_start"],
            "frame_end": e["frame_end"],
            "timestamp_start_sec": e["timestamp_start_sec"],
            "timestamp_end_sec": e["timestamp_end_sec"],
            "duration_sec": e["duration_sec"],
            "peak_location": e["peak_location"],
            "bounding_box": bbox,
            "area_radius_px": None,
            "direction": e.get("direction"),
            "peak_deviation_px": e.get("peak_deviation_px"),
            "peak_angle_diff_deg": None,
            "mean_angle_diff_deg": None,
        })

    for e in angle["error_events"]:
        bbox = compute_angle_bbox(e, frame_checks, padding=80) if corridor_available else None
        all_errors.append({
            "error_id": None,
            "error_type": "angle",
            "frame_start": e["frame_start"],
            "frame_end": e["frame_end"],
            "timestamp_start_sec": e["timestamp_start_sec"],
            "timestamp_end_sec": e["timestamp_end_sec"],
            "duration_sec": e["duration_sec"],
            "peak_location": e["peak_location"],
            "bounding_box": bbox,
            "area_radius_px": None,
            "direction": None,
            "peak_deviation_px": None,
            "peak_angle_diff_deg": e.get("peak_angle_diff_deg"),
            "mean_angle_diff_deg": e.get("mean_angle_diff_deg"),
        })

    # Vibration errors arrive pre-built with bboxes — just append them
    vib_errors: list[dict] = list(vibration_errors) if vibration_errors else []
    for e in vib_errors:
        e["error_id"] = None  # will be reassigned below
        all_errors.append(e)

    # Sort by start timestamp, then assign sequential global IDs
    all_errors.sort(key=lambda e: e["timestamp_start_sec"])
    for idx, e in enumerate(all_errors):
        e["error_id"] = idx + 1

    output = {
        "run_id": traj["run_id"],
        "expert_id": traj.get("expert_id"),
        "total_errors": len(all_errors),
        "trajectory_error_count": len(traj["error_events"]),
        "angle_error_count": len(angle["error_events"]),
        "vibration_error_count": len(vib_errors),
        "all_errors": all_errors,
        "trajectory_errors": traj["error_events"],
        "angle_errors": angle["error_events"],
        "vibration_errors": vib_errors,
    }

    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "unified_errors.json")
    with open(output_path, "w") as fh:
        json.dump(output, fh, indent=2)

    traj_count = len(traj["error_events"])
    angle_count = len(angle["error_events"])
    vib_count = len(vib_errors)
    bbox_label = "with bounding boxes" if corridor_available else "no corridor data"

    print("\n=== UNIFIED ERROR SUMMARY ===")
    print(f"Trajectory errors: {traj_count} ({bbox_label})")
    print(f"Angle errors: {angle_count} ({bbox_label})")
    print(f"Vibration errors: {vib_count}")
    print(f"Total errors: {len(all_errors)}")
    for e in all_errors:
        bb = e["bounding_box"]
        tag = f"[{e['error_type']}]".ljust(14)
        if bb:
            print(
                f"  Error {e['error_id']} {tag} "
                f"bbox ({bb['x_min']}, {bb['y_min']})-({bb['x_max']}, {bb['y_max']})"
            )
        else:
            loc = e["peak_location"]
            print(f"  Error {e['error_id']} {tag} peak at ({loc['x']}, {loc['y']}) (no bbox)")
    print(f"Saved: {output_path}")

    return output
