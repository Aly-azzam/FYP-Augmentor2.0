"""One-time offline expert YOLO+SAM2 scissors preprocessing.

This terminal-only tool generates a reusable expert reference once and saves
it under:

    storage/outputs/sam2_yolo/experts/<expert_code>/

It must not be called from learner upload/evaluation paths. Learner runs should
compare against these saved raw.json/metrics.json artifacts instead of rerunning
the expert video.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.services.sam2_yolo.runner import (
    DEFAULT_FRAME_STRIDE,
    run_sam2_yolo_scissors_tracking,
)
from app.services.sam2_yolo.schemas import (
    METRICS_FILENAME,
    MODEL_NAME,
    OVERLAY_FILENAME,
    PROMPT_SOURCE,
    RAW_FILENAME,
    SUMMARY_FILENAME,
    TRACKING_POINT_TYPE,
)


BACKEND_ROOT = Path(__file__).resolve().parents[2]
EXPERTS_ROOT = BACKEND_ROOT / "storage" / "outputs" / "sam2_yolo" / "experts"
METADATA_FILENAME = "metadata.json"


def process_expert_sam2_yolo(
    *,
    expert_video_path: str | Path,
    expert_code: str,
    stride: int = DEFAULT_FRAME_STRIDE,
    use_gpu: bool = True,
    overwrite: bool = False,
    tracking_point_type: str = TRACKING_POINT_TYPE,
    save_debug: bool = False,
    max_processed_frames: int | None = None,
) -> dict[str, Any]:
    video_path = Path(expert_video_path).expanduser().resolve()
    if not video_path.is_file():
        raise FileNotFoundError(f"expert_video_path does not exist: {video_path}")

    resolved_expert_code = _validate_expert_code(expert_code)
    expert_dir = EXPERTS_ROOT / resolved_expert_code
    existing_payload = _existing_expert_payload(expert_dir, resolved_expert_code)
    if existing_payload is not None and not overwrite:
        existing_payload["reused_existing"] = True
        return existing_payload

    if expert_dir.exists() and overwrite:
        shutil.rmtree(expert_dir)

    summary = run_sam2_yolo_scissors_tracking(
        video_path=video_path,
        output_root=EXPERTS_ROOT,
        frame_stride=stride,
        tracking_point_type=tracking_point_type,
        use_gpu=use_gpu,
        save_debug=save_debug,
        run_id=resolved_expert_code,
        max_processed_frames=max_processed_frames,
    )

    metadata = _write_expert_metadata(
        expert_dir=expert_dir,
        expert_code=resolved_expert_code,
        source_video_path=video_path,
        frame_stride=stride,
        tracking_point_type=tracking_point_type,
        max_processed_frames=max_processed_frames,
    )
    _stamp_expert_payloads(
        expert_dir=expert_dir,
        expert_code=resolved_expert_code,
        metadata=metadata,
    )

    payload = _existing_expert_payload(expert_dir, resolved_expert_code)
    if payload is None:
        raise RuntimeError(f"Expert YOLO+SAM2 output was not created at {expert_dir}")
    payload["reused_existing"] = False
    payload["status"] = summary.get("status", payload.get("status"))
    return payload


def _existing_expert_payload(expert_dir: Path, expert_code: str) -> dict[str, Any] | None:
    raw_path = expert_dir / RAW_FILENAME
    metrics_path = expert_dir / METRICS_FILENAME
    summary_path = expert_dir / SUMMARY_FILENAME
    overlay_path = expert_dir / OVERLAY_FILENAME
    metadata_path = expert_dir / METADATA_FILENAME

    required_paths = [raw_path, metrics_path, summary_path, overlay_path, metadata_path]
    if not all(path.is_file() for path in required_paths):
        return None

    summary = _read_json(summary_path)
    return {
        "expert_code": expert_code,
        "status": summary.get("status", "success"),
        "model": summary.get("model", MODEL_NAME),
        "device": summary.get("device"),
        "frame_stride": summary.get("frame_stride"),
        "raw_json_path": str(raw_path),
        "metrics_json_path": str(metrics_path),
        "summary_json_path": str(summary_path),
        "overlay_video_path": str(overlay_path),
        "metadata_json_path": str(metadata_path),
        "processed_frames": summary.get("processed_frames"),
        "successful_masks": summary.get("successful_masks"),
        "main_tracked_feature": summary.get("main_tracked_feature"),
        "trajectory_stability_score": summary.get("trajectory_stability_score"),
        "horizontal_drift_range": summary.get("horizontal_drift_range"),
        "region_stability_score": summary.get("region_stability_score"),
        "tracking_quality_note": summary.get("tracking_quality_note"),
    }


def _write_expert_metadata(
    *,
    expert_dir: Path,
    expert_code: str,
    source_video_path: Path,
    frame_stride: int,
    tracking_point_type: str,
    max_processed_frames: int | None,
) -> dict[str, Any]:
    metadata = {
        "expert_code": expert_code,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_video_path": str(source_video_path),
        "frame_stride": frame_stride,
        "max_processed_frames": max_processed_frames,
        "tracking_point_type": tracking_point_type,
        "model": MODEL_NAME,
        "prompt_source": PROMPT_SOURCE,
        "raw_json_path": str(expert_dir / RAW_FILENAME),
        "metrics_json_path": str(expert_dir / METRICS_FILENAME),
        "summary_json_path": str(expert_dir / SUMMARY_FILENAME),
        "overlay_video_path": str(expert_dir / OVERLAY_FILENAME),
    }
    _write_json(expert_dir / METADATA_FILENAME, metadata)
    return metadata


def _stamp_expert_payloads(
    *,
    expert_dir: Path,
    expert_code: str,
    metadata: dict[str, Any],
) -> None:
    raw_path = expert_dir / RAW_FILENAME
    metrics_path = expert_dir / METRICS_FILENAME
    summary_path = expert_dir / SUMMARY_FILENAME

    if raw_path.is_file():
        raw = _read_json(raw_path)
        raw["expert_code"] = expert_code
        _write_json(raw_path, raw)

    if metrics_path.is_file():
        metrics = _read_json(metrics_path)
        metrics["expert_code"] = expert_code
        _write_json(metrics_path, metrics)

    if summary_path.is_file():
        summary = _read_json(summary_path)
        summary["expert_code"] = expert_code
        summary["raw_json_path"] = metadata["raw_json_path"]
        summary["metrics_json_path"] = metadata["metrics_json_path"]
        summary["summary_json_path"] = metadata["summary_json_path"]
        summary["overlay_video_path"] = metadata["overlay_video_path"]
        _write_json(summary_path, summary)


def _validate_expert_code(expert_code: str) -> str:
    value = expert_code.strip()
    if not value:
        raise ValueError("--expert_code is required")
    if any(char in value for char in ('/', "\\", ":", "*", "?", '"', "<", ">", "|")):
        raise ValueError(f"--expert_code contains invalid path characters: {expert_code!r}")
    return value


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="process_expert_sam2_yolo",
        description=(
            "Run YOLO+SAM2 scissors preprocessing on a fixed expert video once "
            "and save reusable raw/metrics/summary artifacts."
        ),
    )
    parser.add_argument("--expert_video_path", required=True, help="Path to the expert video.")
    parser.add_argument("--expert_code", required=True, help="Stable expert reference code.")
    parser.add_argument("--stride", type=int, default=DEFAULT_FRAME_STRIDE, help="SAM2 frame stride.")
    parser.add_argument(
        "--max_processed_frames",
        type=int,
        default=0,
        help="Maximum processed JPEG frames SAM2 should load. Use 0 for no cap.",
    )
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
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Rerun YOLO+SAM2 and replace the existing expert output.",
    )
    parser.add_argument(
        "--tracking_point_type",
        choices=[TRACKING_POINT_TYPE],
        default=TRACKING_POINT_TYPE,
        help="Tracking point used for reusable trajectory data.",
    )
    parser.add_argument(
        "--save_debug",
        action="store_true",
        help="Save debug_first_frame.jpg with YOLO bbox and SAM2 prompt point.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    try:
        payload = process_expert_sam2_yolo(
            expert_video_path=args.expert_video_path,
            expert_code=args.expert_code,
            stride=args.stride,
            use_gpu=args.use_gpu,
            overwrite=args.overwrite,
            tracking_point_type=args.tracking_point_type,
            save_debug=args.save_debug,
            max_processed_frames=args.max_processed_frames if args.max_processed_frames > 0 else None,
        )
    except Exception as exc:  # noqa: BLE001 - top-level CLI guard
        print(f"[FAIL] Expert YOLO+SAM2 preprocessing failed: {exc}", file=sys.stderr)
        return 1

    if payload.get("reused_existing"):
        print(f"[OK] Expert YOLO+SAM2 output already exists for {payload['expert_code']}; reused it.")
    else:
        print(f"[OK] Processed expert YOLO+SAM2 output for {payload['expert_code']}.")
    print(json.dumps(payload, indent=2))
    return 0 if payload.get("status") != "failed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
