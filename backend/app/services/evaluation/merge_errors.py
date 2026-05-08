"""Merge trajectory and angle errors into a single unified_errors.json.

Inputs
------
storage/evaluation/{run_id}/trajectory/trajectory_errors.json
storage/evaluation/{run_id}/angle/angle_errors.json

Output
------
storage/evaluation/{run_id}/score/unified_errors.json
"""

from __future__ import annotations

import json
import os
from pathlib import Path


def merge_errors(
    trajectory_errors_path: str,
    angle_errors_path: str,
    output_dir: str,
) -> dict:
    """Merge trajectory and angle error events into one sorted unified list.

    Parameters
    ----------
    trajectory_errors_path:
        Path to ``trajectory_errors.json``.
    angle_errors_path:
        Path to ``angle_errors.json``.
    output_dir:
        Directory where ``unified_errors.json`` is written
        (typically ``storage/evaluation/{run_id}/score/``).

    Returns
    -------
    dict
        The full unified payload that was also written to disk.
    """
    with open(trajectory_errors_path) as fh:
        traj = json.load(fh)
    with open(angle_errors_path) as fh:
        angle = json.load(fh)

    all_errors: list[dict] = []

    for e in traj["error_events"]:
        all_errors.append({
            "error_id": None,            # assigned after sorting
            "error_type": "trajectory",
            "frame_start": e["frame_start"],
            "frame_end": e["frame_end"],
            "timestamp_start_sec": e["timestamp_start_sec"],
            "timestamp_end_sec": e["timestamp_end_sec"],
            "duration_sec": e["duration_sec"],
            "peak_location": e["peak_location"],
            "area_radius_px": None,       # trajectory errors use a path band
            "path_band_width_px": 120,    # wide band for user to click on path
            "direction": e.get("direction"),
            "peak_deviation_px": e.get("peak_deviation_px"),
            "peak_angle_diff_deg": None,
            "mean_angle_diff_deg": None,
        })

    for e in angle["error_events"]:
        all_errors.append({
            "error_id": None,
            "error_type": "angle",
            "frame_start": e["frame_start"],
            "frame_end": e["frame_end"],
            "timestamp_start_sec": e["timestamp_start_sec"],
            "timestamp_end_sec": e["timestamp_end_sec"],
            "duration_sec": e["duration_sec"],
            "peak_location": e["peak_location"],
            "area_radius_px": e["area_radius_px"],
            "path_band_width_px": None,
            "direction": None,
            "peak_deviation_px": None,
            "peak_angle_diff_deg": e.get("peak_angle_diff_deg"),
            "mean_angle_diff_deg": e.get("mean_angle_diff_deg"),
        })

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
        "all_errors": all_errors,
        "trajectory_errors": traj["error_events"],
        "angle_errors": angle["error_events"],
    }

    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "unified_errors.json")
    with open(output_path, "w") as fh:
        json.dump(output, fh, indent=2)

    print("\n=== UNIFIED ERROR SUMMARY ===")
    print(f"Trajectory errors: {len(traj['error_events'])}")
    print(f"Angle errors:      {len(angle['error_events'])}")
    print(f"Total errors:      {len(all_errors)}")
    for e in all_errors:
        print(
            f"  Error {e['error_id']} [{e['error_type']}]: "
            f"frames {e['frame_start']}-{e['frame_end']} | "
            f"{e['timestamp_start_sec']}s-{e['timestamp_end_sec']}s"
        )
    print(f"Saved: {output_path}")

    return output
