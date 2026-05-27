"""SAM2+YOLO pipeline entry point for the evaluation orchestrator.

Provides run_sam2_pipeline_from_yolo — accepts pre-computed YOLO results
from yolo_shared.run_shared_yolo so YOLO never runs twice per evaluation.

The original run_sam2_yolo_scissors_tracking() in runner.py is untouched
and continues to power the Tools-tab "Run YOLO+SAM2" button.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from app.services.sam2_yolo.schemas import (
    SUMMARY_FILENAME,
    InitialPrompt,
)
from app.services.sam2_yolo.yolo_scissors_detector import (
    DEFAULT_BBOX_SHRINK,
    shrink_bbox,
)

logger = logging.getLogger(__name__)

BACKEND_ROOT = Path(__file__).resolve().parents[3]
_EXPERTS_ROOT = BACKEND_ROOT / "storage" / "outputs" / "sam2_yolo" / "experts"


def run_sam2_pipeline_from_yolo(
    *,
    video_path: str,
    yolo_result: dict,
    expert_id: str,
    output_dir: str,
    generate_overlay_video: bool = False,
) -> dict[str, Any]:
    """Run the SAM2 trajectory pipeline using a pre-computed YOLO result.

    Builds an ``InitialPrompt`` from the first valid YOLO detection, then
    runs SAM2 tracking + corridor alignment. YOLO inference is skipped.

    Parameters
    ----------
    video_path:
        Absolute path to the learner video.
    yolo_result:
        Dict returned by ``run_shared_yolo`` — must contain
        ``first_valid_frame``, ``first_valid_bbox``, ``first_valid_bbox_center``,
        and ``all_detections``.
    expert_id:
        Expert code used for corridor alignment (same value as expert_code
        in the Tools-tab pipeline).
    output_dir:
        Directory where all SAM2 outputs are saved (raw.json, summary.json,
        cleaned_trajectory.json, trajectory_smoothed.json,
        aligned_corridor.json, overlay videos, …).

    Returns
    -------
    dict — SAM2 summary with extra normalised keys:
        aligned_corridor_path : str | None
        raw_json_path         : str | None
        trajectory_smoothed_json_path : str | None
    """
    from app.services.sam2_yolo.stable_runner import (
        run_stable_yolo_sam2_tracking_with_prompt,
    )

    # ── Build InitialPrompt from yolo_result ──────────────────────────────────
    first_valid_frame = yolo_result.get("first_valid_frame")
    first_valid_bbox = yolo_result.get("first_valid_bbox")
    first_valid_bbox_center = yolo_result.get("first_valid_bbox_center")

    if first_valid_frame is None or first_valid_bbox is None:
        raise RuntimeError(
            "YOLO did not detect scissors in any frame. "
            "Cannot initialise SAM2 without a valid first bbox."
        )

    # Look up confidence for the first valid frame.
    confidence = 0.5
    for det in yolo_result.get("all_detections", []):
        if det.get("frame_index") == first_valid_frame and det.get("detected"):
            confidence = float(det.get("confidence") or 0.5)
            break

    raw_bbox = [float(v) for v in first_valid_bbox]
    sam2_bbox = shrink_bbox(raw_bbox, shrink_factor=DEFAULT_BBOX_SHRINK)

    if first_valid_bbox_center:
        center_point = [float(v) for v in first_valid_bbox_center]
    else:
        center_point = [
            round((raw_bbox[0] + raw_bbox[2]) / 2.0, 3),
            round((raw_bbox[1] + raw_bbox[3]) / 2.0, 3),
        ]

    # SAM2 re-reads fps from the video; 30.0 is a placeholder for timestamp only.
    timestamp_sec = round(float(first_valid_frame) / 30.0, 6)

    initial_prompt = InitialPrompt(
        frame_index=int(first_valid_frame),
        timestamp_sec=timestamp_sec,
        point=center_point,
        bbox=sam2_bbox,
        raw_yolo_bbox=raw_bbox,
        sam2_prompt_bbox=sam2_bbox,
        bbox_shrink=round(float(DEFAULT_BBOX_SHRINK), 6),
        confidence=round(confidence, 6),
    )

    # ── SAM2 tracking ─────────────────────────────────────────────────────────
    out_dir = Path(output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = run_stable_yolo_sam2_tracking_with_prompt(
        video_path=video_path,
        output_dir=out_dir,
        initial_prompt=initial_prompt,
    )

    # ── Corridor alignment ────────────────────────────────────────────────────
    if expert_id and summary.get("status") == "success":
        summary = _apply_corridor_alignment(
            summary=summary,
            expert_code=expert_id,
            summary_path=out_dir / SUMMARY_FILENAME,
            generate_overlay_video=generate_overlay_video,
        )

    # Normalise key names for the orchestrator.
    summary.setdefault("aligned_corridor_path", summary.get("aligned_corridor_json_path"))
    summary.setdefault("raw_json_path", summary.get("raw_json_path"))

    return summary


def generate_corridor_overlay_for_run(
    *,
    run_id: str,
    eval_base_dir: str,
) -> dict[str, str]:
    """Generate the corridor overlay video on demand for a completed evaluation run.

    Reads all required inputs from the evaluation output directory (no re-computation
    of alignment), calls align_corridor_blade_tip_with_extension with
    generate_overlay_video=True, and returns the overlay video path and URL.

    Parameters
    ----------
    run_id:
        The evaluation run_id (UUID string).
    eval_base_dir:
        Absolute path to the evaluation storage root, i.e.
        ``storage/evaluation/`` — NOT the run folder itself.

    Returns
    -------
    dict with keys:
        overlay_video_path: str
        overlay_video_url: str
    """
    import json as _json  # noqa: PLC0415

    run_dir = Path(eval_base_dir) / run_id
    trajectory_dir = run_dir / "trajectory"

    # ── Resolve required inputs ───────────────────────────────────────────────
    aligned_corridor_json = trajectory_dir / "aligned_corridor.json"
    if not aligned_corridor_json.is_file():
        raise FileNotFoundError(
            f"Corridor data not found for this run. "
            f"Expected: {aligned_corridor_json}"
        )

    # Read the summary to recover expert_code and trajectory paths.
    summary_path = trajectory_dir / "summary.json"
    if not summary_path.is_file():
        raise FileNotFoundError(f"summary.json not found: {summary_path}")
    summary = _json.loads(summary_path.read_text(encoding="utf-8"))

    expert_code: str = summary.get("alignment_expert_code") or summary.get("expert_code", "")
    if not expert_code:
        raise ValueError("Cannot determine expert_code from summary.json")

    learner_run_id: str = summary.get("run_id", run_id)
    trajectory_smoothed_path: str | None = summary.get("trajectory_smoothed_json_path")
    learner_raw_path: str | None = summary.get("raw_json_path")

    # Recover learner video path from yolo_detections.json.
    yolo_json = run_dir / "yolo_detections.json"
    learner_video_path: str | None = None
    if yolo_json.is_file():
        yolo_data = _json.loads(yolo_json.read_text(encoding="utf-8"))
        learner_video_path = yolo_data.get("video_path")

    corridor_path = _EXPERTS_ROOT / expert_code / "corridor.json"
    if not corridor_path.is_file():
        raise FileNotFoundError(
            f"Expert corridor not found. Expected: {corridor_path}"
        )
    expert_raw_path = _EXPERTS_ROOT / expert_code / "raw.json"
    if not expert_raw_path.is_file():
        raise FileNotFoundError(
            f"Expert raw trajectory not found. Expected: {expert_raw_path}"
        )
    if not trajectory_smoothed_path or not Path(trajectory_smoothed_path).is_file():
        raise FileNotFoundError(
            f"trajectory_smoothed.json not found: {trajectory_smoothed_path}"
        )
    if not learner_raw_path or not Path(learner_raw_path).is_file():
        raise FileNotFoundError(
            f"learner raw.json not found: {learner_raw_path}"
        )

    from app.services.sam2_yolo.corridor_alignment import (  # noqa: PLC0415
        align_corridor_blade_tip_with_extension,
        ALIGNED_CORRIDOR_OVERLAY_FILENAME,
    )

    result = align_corridor_blade_tip_with_extension(
        corridor_path=str(corridor_path),
        expert_raw_path=str(expert_raw_path),
        learner_raw_path=str(learner_raw_path),
        learner_smoothed_path=str(trajectory_smoothed_path),
        learner_run_id=learner_run_id,
        output_dir=str(trajectory_dir),
        expert_code=expert_code,
        learner_video_path=learner_video_path,
        generate_overlay_video=True,
    )

    return {
        "overlay_video_path": result["aligned_corridor_overlay_video_path"],
        "overlay_video_url": result["aligned_corridor_overlay_video_url"],
    }


# ── Corridor alignment (local copy of the runner helper) ──────────────────────
# Duplicated here to avoid importing a private function from runner.py.

def _apply_corridor_alignment(
    summary: dict[str, Any],
    expert_code: str,
    summary_path: Path,
    generate_overlay_video: bool = True,
) -> dict[str, Any]:
    from app.services.sam2_yolo.corridor_alignment import (  # noqa: PLC0415
        align_corridor_blade_tip_with_extension,
    )

    trajectory_smoothed_path = summary.get("trajectory_smoothed_json_path")
    if not trajectory_smoothed_path or not Path(trajectory_smoothed_path).is_file():
        logger.warning(
            "[corridor_alignment] trajectory_smoothed.json not found, skipping"
        )
        return summary

    learner_raw_path = summary.get("raw_json_path")
    if not learner_raw_path or not Path(learner_raw_path).is_file():
        logger.warning(
            "[corridor_alignment] learner raw.json not found, skipping"
        )
        return summary

    expert_raw_path = _EXPERTS_ROOT / expert_code / "raw.json"
    if not expert_raw_path.is_file():
        raise FileNotFoundError(
            f"Expert SAM2 raw trajectory not found. "
            f"Run the offline expert script first. "
            f"Expected: {expert_raw_path}"
        )

    corridor_path = _EXPERTS_ROOT / expert_code / "corridor.json"
    if not corridor_path.is_file():
        raise FileNotFoundError(
            f"Expert SAM2 corridor not found. "
            f"Run the offline expert script first. "
            f"Expected: {corridor_path}"
        )

    run_dir = Path(summary["run_dir"])
    learner_video_path: str | None = summary.get("video_path")

    try:
        alignment = align_corridor_blade_tip_with_extension(
            corridor_path=str(corridor_path),
            expert_raw_path=str(expert_raw_path),
            learner_raw_path=learner_raw_path,
            learner_smoothed_path=trajectory_smoothed_path,
            learner_run_id=summary["run_id"],
            output_dir=str(run_dir),
            expert_code=expert_code,
            learner_video_path=learner_video_path,
            generate_overlay_video=generate_overlay_video,
        )
        summary.update(
            {
                "aligned_corridor_json_path": alignment["aligned_corridor_json_path"],
                "aligned_corridor_path": alignment["aligned_corridor_json_path"],
                "aligned_corridor_preview_path": alignment["aligned_corridor_preview_path"],
                "aligned_corridor_preview_url": alignment["aligned_corridor_preview_url"],
                "aligned_corridor_overlay_video_path": alignment[
                    "aligned_corridor_overlay_video_path"
                ],
                "aligned_corridor_overlay_video_url": alignment[
                    "aligned_corridor_overlay_video_url"
                ],
                "alignment_mode": "blade_tip_translation_with_robust_extension",
                "alignment_expert_code": expert_code,
            }
        )
        import json as _json  # noqa: PLC0415
        summary_path.write_text(_json.dumps(summary, indent=2), encoding="utf-8")
    except Exception as exc:  # noqa: BLE001
        logger.exception("[corridor_alignment] Corridor alignment failed")
        summary["aligned_corridor_error"] = str(exc)

    return summary
