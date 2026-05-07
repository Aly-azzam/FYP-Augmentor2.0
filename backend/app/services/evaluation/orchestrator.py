"""Evaluation orchestrator — runs the full learner evaluation pipeline.

Step 1 — YOLO (once):
    run_shared_yolo → yolo_detections.json
    Feeds both SAM2 and the angle pipeline — YOLO never runs twice.

Step 2 — SAM2 trajectory pipeline:
    run_sam2_pipeline_from_yolo → raw.json, cleaned_trajectory.json,
    trajectory_smoothed.json, aligned_corridor.json

    detect_trajectory_errors → trajectory_errors.json

Step 3 — Angle + DTW pipeline:
    run_learner_angle_from_yolo → angles.json, dtw_alignment.json,
    learner_angle_comparison.json, run_summary.json,
    learner_comparison_output.mp4

Output layout:
    storage/evaluation/{run_id}/
        yolo_detections.json
        trajectory/
            raw.json
            cleaned_trajectory.json
            trajectory_smoothed.json
            aligned_corridor.json
            trajectory_errors.json
            summary.json
            …overlay videos…
        angle/
            angles.json
            dtw_alignment.json
            learner_angle_comparison.json
            run_summary.json
            learner_comparison_output.mp4
        score/
        visualization/
        vlm/
"""

from __future__ import annotations

import os
import time
import uuid
from pathlib import Path
from typing import Any, Callable

_BACKEND_ROOT = Path(__file__).resolve().parents[3]


def run_evaluation_pipeline(
    learner_video_path: str,
    expert_id: str,
    emit_progress: Callable[[str], None],
) -> dict[str, Any]:
    """Run the full learner evaluation pipeline.

    Parameters
    ----------
    learner_video_path:
        Absolute path to the learner video file.
    expert_id:
        Chapter UUID / expert code used for corridor alignment and angle
        comparison.  Same value passed as ``expert_name`` / ``expert_code``
        in the individual pipelines.
    emit_progress:
        Callable accepting a step-name string.  Called before each step so
        callers (SSE endpoints, tests, …) can stream progress.

    Returns
    -------
    dict with keys:
        run_id, dirs, yolo_detections_path,
        trajectory_errors_path, dtw_alignment_path, status
    """
    run_id = str(uuid.uuid4())
    t_start = time.time()

    # ── Output directories — created all upfront ──────────────────────────────
    base_dir = str(_BACKEND_ROOT / "storage" / "evaluation" / run_id)
    dirs: dict[str, str] = {
        "root":          base_dir,
        "trajectory":    f"{base_dir}/trajectory",
        "angle":         f"{base_dir}/angle",
        "score":         f"{base_dir}/score",
        "visualization": f"{base_dir}/visualization",
        "vlm":           f"{base_dir}/vlm",
    }
    for d in dirs.values():
        os.makedirs(d, exist_ok=True)

    # ── Step 1: YOLO — runs once, feeds both pipelines ────────────────────────
    emit_progress("yolo")
    from app.services.evaluation.yolo_shared import run_shared_yolo  # noqa: PLC0415

    t0 = time.time()
    yolo_result = run_shared_yolo(
        video_path=learner_video_path,
        output_dir=dirs["root"],
    )
    print(f"[TIMING] YOLO: {time.time() - t0:.1f}s")
    # saves: storage/evaluation/{run_id}/yolo_detections.json
    print("[EVALUATE] Step 1/3 — YOLO detection complete")

    # ── Step 2: SAM2 trajectory pipeline ─────────────────────────────────────
    emit_progress("trajectory_init")
    from app.services.sam2_yolo.pipeline import run_sam2_pipeline_from_yolo  # noqa: PLC0415

    t0 = time.time()
    sam2_result = run_sam2_pipeline_from_yolo(
        video_path=learner_video_path,
        yolo_result=yolo_result,
        expert_id=expert_id,
        output_dir=dirs["trajectory"],
        generate_overlay_video=False,  # generated on demand only
    )
    print(f"[TIMING] SAM2 pipeline: {time.time() - t0:.1f}s")
    # saves to trajectory/:
    #   raw.json, cleaned_trajectory.json,
    #   trajectory_smoothed.json, aligned_corridor.json

    emit_progress("trajectory_track")

    emit_progress("trajectory_errors")
    from app.services.sam2_yolo.trajectory_errors import detect_trajectory_errors  # noqa: PLC0415

    aligned_corridor_path = sam2_result.get("aligned_corridor_path") or sam2_result.get(
        "aligned_corridor_json_path"
    )
    raw_json_path = sam2_result.get("raw_json_path")

    trajectory_errors: dict[str, Any] = {}
    if aligned_corridor_path and Path(aligned_corridor_path).is_file() and raw_json_path:
        t0 = time.time()
        trajectory_errors = detect_trajectory_errors(
            aligned_corridor_path=str(aligned_corridor_path),
            raw_json_path=str(raw_json_path),
            output_dir=dirs["trajectory"],
        )
        print(f"[TIMING] Trajectory errors: {time.time() - t0:.1f}s")
        # saves to trajectory/: trajectory_errors.json
    else:
        print(
            "[EVALUATE] WARNING: aligned_corridor.json not found — "
            "skipping trajectory error detection"
        )

    print("[EVALUATE] Step 2/3 — SAM2 trajectory pipeline complete")

    # ── Step 3: Angle + DTW pipeline ─────────────────────────────────────────
    emit_progress("angle_init")
    from app.services.angle.learner_pipeline import run_learner_angle_from_yolo  # noqa: PLC0415

    t0 = time.time()
    angle_result = run_learner_angle_from_yolo(
        video_path=learner_video_path,
        yolo_detections=yolo_result["all_detections"],
        expert_name=expert_id,
        output_dir=dirs["angle"],
    )
    print(f"[TIMING] Angle pipeline: {time.time() - t0:.1f}s")
    # saves to angle/:
    #   angles.json, dtw_alignment.json,
    #   learner_angle_comparison.json, run_summary.json,
    #   learner_comparison_output.mp4

    emit_progress("angle_track")
    emit_progress("done")

    print("[EVALUATE] Step 3/3 — Angle + DTW pipeline complete")
    print(f"[TIMING] Total: {time.time() - t_start:.1f}s")
    print(f"[EVALUATE] Run complete — run_id: {run_id}")

    return {
        "run_id": run_id,
        "dirs": dirs,
        "yolo_detections_path": f"{dirs['root']}/yolo_detections.json",
        "trajectory_errors_path": f"{dirs['trajectory']}/trajectory_errors.json",
        "dtw_alignment_path": f"{dirs['angle']}/dtw_alignment.json",
        "sam2_result": {
            "status": sam2_result.get("status"),
            "raw_json_path": raw_json_path,
            "aligned_corridor_path": aligned_corridor_path,
            "trajectory_smoothed_json_path": sam2_result.get("trajectory_smoothed_json_path"),
        },
        "angle_result": {
            "run_id": angle_result.get("run_id"),
            "normalized_dtw_distance": angle_result.get("normalized_dtw_distance"),
            "mean_angle_difference": angle_result.get("mean_angle_difference"),
            "dtw_alignment_path": angle_result.get("dtw_alignment_path"),
            "learner_comparison_video_path": angle_result.get("learner_comparison_video_path"),
        },
        "status": "done",
    }
