from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
import uuid
from pathlib import Path
from typing import Any

from app.services.sam2_yolo.region_metrics import compute_region_metrics
from app.services.sam2_yolo.schemas import (
    DEFAULT_TRAJECTORY_POINT_MODE,
    METRICS_FILENAME,
    MODEL_NAME,
    OVERLAY_FILENAME,
    PROMPT_SOURCE,
    RAW_FILENAME,
    SUMMARY_FILENAME,
    TRACKING_POINT_TYPE,
    TRACKING_QUALITY_NOTE,
    TRAJECTORY_POINT_MODES,
    regions_from_frames,
    trajectory_from_frames,
)
from app.services.sam2_yolo.trajectory_metrics import compute_trajectory_metrics
from app.services.sam2_yolo.yolo_scissors_detector import (
    DEFAULT_ROBOFLOW_CONFIDENCE,
)
from app.services.sam2_yolo.stable_runner import run_stable_yolo_sam2_run


BACKEND_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_FRAME_STRIDE = 5
DEFAULT_OUTPUT_ROOT = BACKEND_ROOT / "storage" / "outputs" / "sam2_yolo" / "runs"
EXPERTS_ROOT = BACKEND_ROOT / "storage" / "outputs" / "sam2_yolo" / "experts"
DEFAULT_MAX_PROCESSED_FRAMES: int | None = None
YOLO_FAILURE_REASON = "YOLO did not detect scissors in the scanned prompt frames"
DEFAULT_RUN_MODE = os.getenv("SAM2_RUN_MODE", "subprocess").strip().lower()
logger = logging.getLogger(__name__)


def run_sam2_yolo_scissors_tracking(
    *,
    video_path: str | Path,
    output_root: str | Path = DEFAULT_OUTPUT_ROOT,
    frame_stride: int = DEFAULT_FRAME_STRIDE,
    tracking_point_type: str = TRACKING_POINT_TYPE,
    trajectory_point_mode: str = DEFAULT_TRAJECTORY_POINT_MODE,
    use_gpu: bool = True,
    save_debug: bool = False,
    run_id: str | None = None,
    max_processed_frames: int | None = None,
    roboflow_confidence: float = DEFAULT_ROBOFLOW_CONFIDENCE,
    sam2_checkpoint: str | None = None,
    sam2_config: str | None = None,
    sam2_model_id: str | None = None,
    expert_code: str | None = None,
) -> dict[str, Any]:
    video = Path(video_path).expanduser().resolve()
    if not video.is_file():
        raise FileNotFoundError(f"video_path does not exist: {video}")
    if trajectory_point_mode not in TRAJECTORY_POINT_MODES:
        raise ValueError(f"trajectory_point_mode must be one of {TRAJECTORY_POINT_MODES}")

    resolved_run_id = run_id or str(uuid.uuid4())
    run_dir = Path(output_root).expanduser().resolve() / resolved_run_id
    if run_dir.exists():
        raise FileExistsError(f"SAM2+YOLO output already exists: {run_dir}")

    frame_cap = max_processed_frames if max_processed_frames is not None else None
    summary_path = run_dir / SUMMARY_FILENAME

    if DEFAULT_RUN_MODE == "direct":
        summary = run_stable_yolo_sam2_run(
            video_path=video,
            output_root=output_root,
            run_id=resolved_run_id,
            stride=frame_stride,
            max_processed_frames=frame_cap,
            use_gpu=use_gpu,
            tracking_point_type=tracking_point_type,
            trajectory_point_mode=trajectory_point_mode,
            save_debug=save_debug,
            roboflow_confidence=roboflow_confidence,
            sam2_checkpoint=sam2_checkpoint,
            sam2_config=sam2_config,
            sam2_model_id=sam2_model_id,
        )
    else:
        run_dir.mkdir(parents=True, exist_ok=False)

        command = [
            sys.executable,
            "-m",
            "app.services.sam2_yolo.stable_runner",
            "--video_path",
            str(video),
            "--output_dir",
            str(run_dir),
            "--stride",
            str(frame_stride),
            "--tracking_point_type",
            tracking_point_type,
            "--trajectory_point_mode",
            trajectory_point_mode,
            "--roboflow_confidence",
            str(roboflow_confidence),
        ]
        if frame_cap is not None:
            command.extend(["--max_processed_frames", str(frame_cap if frame_cap > 0 else 0)])
        if not use_gpu:
            command.append("--no-gpu")
        if save_debug:
            command.append("--save_debug")
        if sam2_checkpoint:
            command.extend(["--sam2_checkpoint", sam2_checkpoint])
        if sam2_config:
            command.extend(["--sam2_config", sam2_config])
        if sam2_model_id:
            command.extend(["--sam2_model_id", sam2_model_id])

        try:
            completed = subprocess.run(  # noqa: S603 - argv is controlled by this backend
                command,
                cwd=BACKEND_ROOT,
                capture_output=True,
                text=True,
                timeout=900,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            failure_reason = f"stable_runner timed out after {exc.timeout} seconds"
            timeout_summary = _build_failed_summary(
                run_id=resolved_run_id,
                video_path=video,
                run_dir=run_dir,
                raw_path=run_dir / RAW_FILENAME,
                metrics_path=run_dir / METRICS_FILENAME,
                summary_path=summary_path,
                overlay_path=run_dir / OVERLAY_FILENAME,
                frame_stride=frame_stride,
                device="unknown",
                device_warning=None,
                failure_reason=failure_reason,
            )
            timeout_summary["stage"] = "sam2_subprocess"
            timeout_summary["device_attempted"] = "unknown"
            timeout_summary["fallback_attempted"] = False
            timeout_summary["max_processed_frames"] = frame_cap
            _write_json(summary_path, timeout_summary)
            return timeout_summary

        if completed.stdout:
            print(completed.stdout, end="", flush=True)
        if completed.stderr:
            print(completed.stderr, end="", file=sys.stderr, flush=True)

        if summary_path.is_file():
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
        else:
            failure_reason = (
                completed.stderr.strip()
                or completed.stdout.strip()
                or f"stable_runner exited with code {completed.returncode}"
            )
            fail_summary = _build_failed_summary(
                run_id=resolved_run_id,
                video_path=video,
                run_dir=run_dir,
                raw_path=run_dir / RAW_FILENAME,
                metrics_path=run_dir / METRICS_FILENAME,
                summary_path=summary_path,
                overlay_path=run_dir / OVERLAY_FILENAME,
                frame_stride=frame_stride,
                device="unknown",
                device_warning=None,
                failure_reason=failure_reason[-1200:],
            )
            fail_summary["stage"] = "sam2_subprocess"
            fail_summary["device_attempted"] = "unknown"
            fail_summary["fallback_attempted"] = False
            fail_summary["max_processed_frames"] = frame_cap
            _write_json(summary_path, fail_summary)
            return fail_summary

    # ── Post-processing: corridor alignment ───────────────────────────────
    if expert_code and summary.get("status") == "success":
        summary = _apply_corridor_alignment(
            summary=summary,
            expert_code=expert_code,
            summary_path=run_dir / SUMMARY_FILENAME,
        )

    return summary


def _apply_corridor_alignment(
    summary: dict[str, Any],
    expert_code: str,
    summary_path: Path,
) -> dict[str, Any]:
    """Run blade-tip translation + robust extension corridor alignment and merge into summary."""
    from app.services.sam2_yolo.corridor_alignment import (  # noqa: PLC0415
        align_corridor_blade_tip_with_extension,
    )

    trajectory_smoothed_path = summary.get("trajectory_smoothed_json_path")
    if not trajectory_smoothed_path or not Path(trajectory_smoothed_path).is_file():
        logger.warning(
            "[corridor_alignment] trajectory_smoothed.json not found, skipping alignment"
        )
        return summary

    learner_raw_path = summary.get("raw_json_path")
    if not learner_raw_path or not Path(learner_raw_path).is_file():
        logger.warning(
            "[corridor_alignment] learner raw.json not found, skipping alignment"
        )
        return summary

    expert_raw_path = EXPERTS_ROOT / expert_code / "raw.json"
    if not expert_raw_path.is_file():
        logger.warning(
            "[corridor_alignment] Expert raw.json not found: %s", expert_raw_path
        )
        summary["aligned_corridor_warning"] = (
            f"Expert raw.json not found for expert_code='{expert_code}'"
        )
        return summary

    corridor_path = EXPERTS_ROOT / expert_code / "corridor.json"
    if not corridor_path.is_file():
        logger.warning(
            "[corridor_alignment] Expert corridor not found: %s", corridor_path
        )
        summary["aligned_corridor_warning"] = (
            f"Expert corridor not found for expert_code='{expert_code}'"
        )
        return summary

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
        )
        summary.update(
            {
                "aligned_corridor_json_path": alignment["aligned_corridor_json_path"],
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
        _write_json(summary_path, summary)
    except Exception as exc:  # noqa: BLE001
        logger.exception("[corridor_alignment] Corridor alignment failed")
        summary["aligned_corridor_error"] = str(exc)

    return summary


def _build_raw_payload(
    *,
    run_id: str,
    video_path: Path,
    device: str,
    frame_stride: int,
    initial_prompt: InitialPrompt,
    frames: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "run_id": run_id,
        "video_path": str(video_path),
        "model": MODEL_NAME,
        "device": device,
        "frame_stride": frame_stride,
        "prompt_source": PROMPT_SOURCE,
        "initial_prompt": initial_prompt.to_dict(),
        "tracking_point_type": TRACKING_POINT_TYPE,
        "frames": frames,
        "trajectory": trajectory_from_frames(frames),
        "regions": regions_from_frames(frames),
    }


def _build_metrics_payload(raw_payload: dict[str, Any]) -> dict[str, Any]:
    trajectory_metrics = compute_trajectory_metrics(raw_payload)
    region_metrics = compute_region_metrics(raw_payload)
    quality_flags = {
        "has_sudden_jumps": trajectory_metrics["jump_count"] > 0,
        "has_lost_masks": trajectory_metrics["lost_tracking_count"] > 0,
        "has_unstable_region": region_metrics["region_stability_score"] < 0.5,
        "usable_for_trajectory_comparison": (
            trajectory_metrics["trajectory_stability_score"] >= 0.25
            and trajectory_metrics["lost_tracking_count"] < len(raw_payload["frames"])
        ),
        "usable_for_region_comparison": region_metrics["region_stability_score"] >= 0.25,
    }
    return {
        "run_id": raw_payload["run_id"],
        "model": MODEL_NAME,
        "frame_stride": raw_payload["frame_stride"],
        "tracking_point_type": TRACKING_POINT_TYPE,
        "trajectory_metrics": trajectory_metrics,
        "region_metrics": region_metrics,
        "quality_flags": quality_flags,
    }


def _build_success_summary(
    *,
    raw_payload: dict[str, Any],
    metrics_payload: dict[str, Any],
    run_dir: Path,
    raw_path: Path,
    metrics_path: Path,
    summary_path: Path,
    overlay_path: Path,
    device_warning: str | None,
) -> dict[str, Any]:
    frames = raw_payload["frames"]
    failure_frames = [frame["frame_index"] for frame in frames if not frame["tracking_valid"]]
    trajectory_metrics = metrics_payload["trajectory_metrics"]
    region_metrics = metrics_payload["region_metrics"]
    summary = {
        "run_id": raw_payload["run_id"],
        "status": "success",
        "model": MODEL_NAME,
        "device": raw_payload["device"],
        "frame_stride": raw_payload["frame_stride"],
        "video_path": raw_payload["video_path"],
        "raw_json_path": str(raw_path),
        "metrics_json_path": str(metrics_path),
        "summary_json_path": str(summary_path),
        "overlay_video_path": str(overlay_path),
        "run_dir": str(run_dir),
        "processed_frames": len(frames),
        "successful_masks": sum(1 for frame in frames if frame["tracking_valid"]),
        "failure_frames": failure_frames,
        "prompt_source": PROMPT_SOURCE,
        "main_tracked_feature": "bbox_center_trajectory",
        "trajectory_stability_score": trajectory_metrics["trajectory_stability_score"],
        "horizontal_drift_range": trajectory_metrics["horizontal_drift_range"],
        "region_stability_score": region_metrics["region_stability_score"],
        "tracking_quality_note": TRACKING_QUALITY_NOTE,
    }
    if device_warning is not None:
        summary["device_warning"] = device_warning
    return summary


def _build_failed_summary(
    *,
    run_id: str,
    video_path: Path,
    run_dir: Path,
    raw_path: Path,
    metrics_path: Path,
    summary_path: Path,
    overlay_path: Path,
    frame_stride: int,
    device: str,
    device_warning: str | None,
    failure_reason: str,
) -> dict[str, Any]:
    summary = {
        "run_id": run_id,
        "status": "failed",
        "failure_reason": failure_reason,
        "model": MODEL_NAME,
        "device": device,
        "frame_stride": frame_stride,
        "video_path": str(video_path),
        "raw_json_path": str(raw_path),
        "metrics_json_path": str(metrics_path),
        "summary_json_path": str(summary_path),
        "overlay_video_path": str(overlay_path),
        "run_dir": str(run_dir),
        "processed_frames": 0,
        "successful_masks": 0,
        "failure_frames": [],
        "prompt_source": PROMPT_SOURCE,
        "main_tracked_feature": "bbox_center_trajectory",
        "tracking_quality_note": TRACKING_QUALITY_NOTE,
    }
    if device_warning is not None:
        summary["device_warning"] = device_warning
    return summary


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Track scissors with Roboflow YOLO and SAM2 using the YOLO bbox center prompt."
    )
    parser.add_argument("--video_path", required=True, help="Path to the input video.")
    parser.add_argument("--stride", type=int, default=DEFAULT_FRAME_STRIDE, help="SAM2 frame stride.")
    parser.add_argument(
        "--use_gpu",
        action="store_true",
        default=True,
        help="Use CUDA when available. Enabled by default.",
    )
    parser.add_argument(
        "--no_gpu",
        action="store_false",
        dest="use_gpu",
        help="Force CPU execution.",
    )
    parser.add_argument("--output_root", default=str(DEFAULT_OUTPUT_ROOT), help="Runs output root.")
    parser.add_argument(
        "--tracking_point_type",
        choices=[TRACKING_POINT_TYPE],
        default=TRACKING_POINT_TYPE,
    )
    parser.add_argument("--save_debug", action="store_true", help="Save debug_first_frame.jpg.")
    parser.add_argument("--run_id", default=None, help="Optional run id. Defaults to a UUID.")
    parser.add_argument(
        "--max_processed_frames",
        type=int,
        default=0,
        help="Maximum processed JPEG frames SAM2 should load. Use 0 for no cap.",
    )
    parser.add_argument(
        "--roboflow_confidence",
        type=float,
        default=DEFAULT_ROBOFLOW_CONFIDENCE,
        help="Roboflow confidence threshold.",
    )
    parser.add_argument("--sam2_checkpoint", default=None, help="Optional local SAM2 checkpoint path.")
    parser.add_argument("--sam2_config", default=None, help="Optional local SAM2 config path/name.")
    parser.add_argument("--sam2_model_id", default=None, help="Optional Hugging Face SAM2 model id.")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    summary = run_sam2_yolo_scissors_tracking(
        video_path=args.video_path,
        output_root=args.output_root,
        frame_stride=args.stride,
        tracking_point_type=args.tracking_point_type,
        use_gpu=args.use_gpu,
        save_debug=args.save_debug,
        run_id=args.run_id,
        max_processed_frames=args.max_processed_frames if args.max_processed_frames > 0 else None,
        roboflow_confidence=args.roboflow_confidence,
        sam2_checkpoint=args.sam2_checkpoint,
        sam2_config=args.sam2_config,
        sam2_model_id=args.sam2_model_id,
    )
    print(json.dumps(summary, indent=2))
    if summary.get("status") == "failed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
