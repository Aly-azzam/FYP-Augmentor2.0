from __future__ import annotations

import argparse
import json
import os
import sys
import uuid
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path


BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

# Optical Flow does not need the DB; provide a safe fallback for imports.
os.environ.setdefault("DATABASE_URL", "sqlite:///./augmentor.db")

from app.services.optical_flow import (  # noqa: E402
    FarnebackConfig,
    build_video_flow_summary,
    compute_video_optical_flow_features,
)
from app.services.optical_flow.visualizer import visualize_video_optical_flow_hsv  # noqa: E402


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Preprocess one expert video into expert-only Optical Flow artifacts.",
    )
    parser.add_argument(
        "--expert-video-id",
        required=True,
        help="Expert video folder ID under storage/expert/.",
    )
    parser.add_argument(
        "--video-path",
        help="Optional explicit expert video path. Relative paths resolve from backend root.",
    )
    parser.add_argument(
        "--save-visualization",
        action="store_true",
        help="Also save an HSV optical flow visualization video.",
    )
    parser.add_argument(
        "--use-hand-roi",
        action="store_true",
        help="Run Farneback only inside detected MediaPipe hand ROIs when possible.",
    )
    parser.add_argument(
        "--roi-padding-px",
        type=int,
        default=40,
        help="Padding in pixels around the detected hand ROI.",
    )
    return parser


def _resolve_path(raw_path: str) -> Path:
    path = Path(raw_path)
    if path.is_absolute():
        return path
    return (BACKEND_ROOT / path).resolve()


def _find_expert_video(expert_video_id: str) -> Path:
    expert_dir = BACKEND_ROOT / "storage" / "expert" / expert_video_id
    if not expert_dir.exists() or not expert_dir.is_dir():
        raise FileNotFoundError(f"Expert video folder not found: {expert_dir}")

    candidates = sorted(
        path
        for path in expert_dir.iterdir()
        if path.is_file() and path.suffix.lower() == ".mp4"
    )
    if not candidates:
        raise FileNotFoundError(f"No .mp4 video found in: {expert_dir}")
    return candidates[0]


def _build_output_dirs(expert_video_id: str) -> tuple[Path, Path, Path]:
    output_root = (
        BACKEND_ROOT
        / "storage"
        / "expert"
        / expert_video_id
        / "optical_flow"
    )
    raw_dir = output_root / "raw"
    processed_dir = output_root / "processed"
    outputs_dir = output_root / "outputs"

    raw_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)
    outputs_dir.mkdir(parents=True, exist_ok=True)
    return raw_dir, processed_dir, outputs_dir


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    expert_video_id = str(args.expert_video_id)
    video_path = (
        _resolve_path(args.video_path)
        if args.video_path
        else _find_expert_video(expert_video_id)
    )
    if not video_path.exists() or not video_path.is_file():
        raise FileNotFoundError(f"Expert video not found: {video_path}")

    raw_dir, processed_dir, outputs_dir = _build_output_dirs(expert_video_id)
    raw_json_path = raw_dir / "expert_optical_flow_raw.json"
    summary_json_path = processed_dir / "expert_optical_flow_summary.json"
    visualization_video_path = outputs_dir / "expert_optical_flow_hsv.mp4"

    config = FarnebackConfig(
        use_hand_roi=args.use_hand_roi,
        roi_padding_px=args.roi_padding_px,
    )
    run_id = f"expert_of_{expert_video_id}_{uuid.uuid4().hex[:12]}"
    created_at = datetime.now(timezone.utc).isoformat()

    video_metadata, frames = compute_video_optical_flow_features(
        video_path=video_path,
        config=config,
    )
    summary = build_video_flow_summary(
        frames,
        roi_enabled=config.use_hand_roi,
    )

    common_payload = {
        "run_id": run_id,
        "expert_video_id": expert_video_id,
        "created_at": created_at,
        "video_metadata": video_metadata.model_dump(mode="json"),
        "summary": summary.model_dump(mode="json"),
        "config_used": asdict(config),
    }
    raw_payload = {
        **common_payload,
        "frames": [frame.model_dump(mode="json") for frame in frames],
    }
    summary_payload = common_payload

    raw_json_path.write_text(
        json.dumps(raw_payload, indent=2),
        encoding="utf-8",
    )
    summary_json_path.write_text(
        json.dumps(summary_payload, indent=2),
        encoding="utf-8",
    )

    created_visualization_path: Path | None = None
    if args.save_visualization:
        created_visualization_path = visualize_video_optical_flow_hsv(
            video_path=video_path,
            output_video_path=visualization_video_path,
            config=config,
            overlay_text="expert",
        )

    print(f"expert_video_id:          {expert_video_id}")
    print(f"video_path:               {video_path}")
    print(f"raw_json_path:            {raw_json_path}")
    print(f"summary_json_path:        {summary_json_path}")
    if created_visualization_path is not None:
        print(f"visualization_video_path: {created_visualization_path}")
    print(f"avg_magnitude:           {summary.avg_magnitude}")
    print(f"motion_stability_score:  {summary.motion_stability_score}")
    print(f"vibration_score:         {summary.vibration_score}")
    print(f"vibration_high_freq_mean:{summary.vibration_high_freq_mean}")
    print(f"roi_usage_ratio:         {summary.roi_usage_ratio}")


if __name__ == "__main__":
    main()
