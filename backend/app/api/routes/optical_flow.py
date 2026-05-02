from __future__ import annotations

import json
import shutil
import subprocess
import time
import uuid
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fastapi import APIRouter, File, Form, HTTPException, UploadFile, status

from app.core.config import settings
from app.services.optical_flow import (
    FarnebackConfig,
    build_video_flow_summary,
    compute_video_optical_flow_features,
)
from app.services.optical_flow.farneback_service import _effective_roi_source
from app.services.optical_flow.visualizer import visualize_video_optical_flow_hsv


router = APIRouter(prefix="/api/optical-flow", tags=["Optical Flow"])


def _resolve_ffmpeg_executable() -> str | None:
    ffmpeg_path = shutil.which("ffmpeg")
    if ffmpeg_path:
        return ffmpeg_path

    try:
        import imageio_ffmpeg
    except ImportError:
        return None

    return str(imageio_ffmpeg.get_ffmpeg_exe())


def _storage_url(path: Path | None) -> str | None:
    if path is None:
        return None

    try:
        relative = path.resolve().relative_to(settings.STORAGE_ROOT.resolve())
    except ValueError:
        return None

    return "/storage/" + relative.as_posix()


def _make_browser_compatible_mp4(video_path: Path) -> None:
    """
    Re-encode OpenCV's mp4v output to a browser-friendly H.264/yuv420p MP4.
    Docker installs ffmpeg; local development can use imageio-ffmpeg.
    """
    ffmpeg_path = _resolve_ffmpeg_executable()
    if ffmpeg_path is None or not video_path.exists():
        return

    temp_path = video_path.with_name(f"{video_path.stem}_browser.mp4")
    command = [
        ffmpeg_path,
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(video_path),
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(temp_path),
    ]
    try:
        subprocess.run(command, check=True)
    except subprocess.CalledProcessError:
        temp_path.unlink(missing_ok=True)
        return

    temp_path.replace(video_path)


@router.get("/health")
async def optical_flow_health() -> dict[str, str]:
    return {"status": "ok"}


@router.post("/learner")
async def process_learner_optical_flow(
    file: UploadFile = File(...),
    save_visualization: bool = Form(default=True),
    use_hand_roi: bool = Form(default=True),
    roi_source: str = Form(default="mediapipe_hand"),
    roi_padding_px: int = Form(default=40),
    max_roi_hold_frames: int = Form(default=5),
    roi_smoothing_enabled: bool = Form(default=True),
    roi_smoothing_alpha: float = Form(default=0.65),
) -> dict[str, Any]:
    """
    Upload and process a learner video with Optical Flow only.

    This endpoint does not compare against an expert video and does not affect
    the main evaluation score or persistence models.
    """
    request_start = time.perf_counter()
    print("[OF] learner request received", flush=True)
    print(f"[OF] roi_source={roi_source}", flush=True)
    print(f"[OF] use_hand_roi={use_hand_roi}", flush=True)
    print(f"[OF] roi_padding_px={roi_padding_px}", flush=True)
    print(f"[OF] save_visualization={save_visualization}", flush=True)

    run_id = f"learner_of_{uuid.uuid4().hex[:12]}"
    learner_video_path = (
        settings.STORAGE_ROOT
        / "uploads"
        / "optical_flow_learner"
        / run_id
        / "learner_video.mp4"
    )
    output_root = settings.STORAGE_ROOT / "learner" / run_id / "optical_flow"
    raw_dir = output_root / "raw"
    processed_dir = output_root / "processed"
    outputs_dir = output_root / "outputs"

    raw_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)
    outputs_dir.mkdir(parents=True, exist_ok=True)
    learner_video_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        await _save_uploaded_video(file, learner_video_path)

        config = FarnebackConfig(
            use_hand_roi=use_hand_roi,
            roi_source=roi_source,  # type: ignore[arg-type]
            roi_padding_px=roi_padding_px,
            max_roi_hold_frames=max_roi_hold_frames,
            roi_smoothing_enabled=roi_smoothing_enabled,
            roi_smoothing_alpha=roi_smoothing_alpha,
        )
        active_roi_source = _effective_roi_source(config)
        created_at = datetime.now(timezone.utc).isoformat()
        video_metadata, frames = compute_video_optical_flow_features(
            video_path=learner_video_path,
            config=config,
        )
        summary = build_video_flow_summary(
            frames,
            roi_enabled=active_roi_source != "none",
            roi_source_used=active_roi_source,
            roi_smoothing_enabled=(
                config.roi_smoothing_enabled
                if active_roi_source in {"yolo_scissors", "yolo_scissors_expanded"}
                else False
            ),
            roi_smoothing_alpha=(
                config.roi_smoothing_alpha
                if active_roi_source in {"yolo_scissors", "yolo_scissors_expanded"}
                else None
            ),
        )

        raw_json_path = raw_dir / "learner_optical_flow_raw.json"
        summary_json_path = processed_dir / "learner_optical_flow_summary.json"
        visualization_video_path = outputs_dir / "learner_optical_flow_hsv.mp4"

        created_visualization_path: Path | None = None
        if save_visualization:
            print("[OF] visualization started", flush=True)
            created_visualization_path = visualize_video_optical_flow_hsv(
                video_path=learner_video_path,
                output_video_path=visualization_video_path,
                config=config,
                overlay_text="learner",
                frame_features=frames,
            )
            _make_browser_compatible_mp4(created_visualization_path)
            print(f"[OF] visualization finished: {created_visualization_path}", flush=True)

        common_payload = {
            "run_id": run_id,
            "created_at": created_at,
            "learner_video_path": str(learner_video_path),
            "learner_video_url": _storage_url(learner_video_path),
            "raw_json_path": str(raw_json_path),
            "raw_json_url": _storage_url(raw_json_path),
            "summary_json_path": str(summary_json_path),
            "summary_json_url": _storage_url(summary_json_path),
            "visualization_video_path": (
                str(created_visualization_path)
                if created_visualization_path is not None
                else None
            ),
            "visualization_video_url": _storage_url(created_visualization_path),
            "video_metadata": video_metadata.model_dump(mode="json"),
            "summary": summary.model_dump(mode="json"),
            "config_used": asdict(config),
        }
        raw_payload = {
            **common_payload,
            "frames": [frame.model_dump(mode="json") for frame in frames],
        }

        print(f"[OF] saving raw JSON: {raw_json_path}", flush=True)
        raw_json_path.write_text(
            json.dumps(raw_payload, indent=2),
            encoding="utf-8",
        )
        print(f"[OF] saving summary JSON: {summary_json_path}", flush=True)
        summary_json_path.write_text(
            json.dumps(common_payload, indent=2),
            encoding="utf-8",
        )
        print("[OF] JSON saved", flush=True)
        total_processing_time_sec = time.perf_counter() - request_start
        print("[OF] learner Optical Flow finished successfully", flush=True)
        print(
            f"[OF] total_processing_time_sec={total_processing_time_sec:.2f}",
            flush=True,
        )

        return {
            "run_id": run_id,
            "learner_video_path": str(learner_video_path),
            "learner_video_url": _storage_url(learner_video_path),
            "raw_json_path": str(raw_json_path),
            "raw_json_url": _storage_url(raw_json_path),
            "summary_json_path": str(summary_json_path),
            "summary_json_url": _storage_url(summary_json_path),
            "visualization_video_path": (
                str(created_visualization_path)
                if created_visualization_path is not None
                else None
            ),
            "visualization_video_url": _storage_url(created_visualization_path),
            "summary": {
                "avg_magnitude": summary.avg_magnitude,
                "motion_stability_score": summary.motion_stability_score,
                "vibration_score": summary.vibration_score,
                "vibration_high_freq_mean": summary.vibration_high_freq_mean,
                "magnitude_jitter": summary.magnitude_jitter,
                "roi_usage_ratio": summary.roi_usage_ratio,
            },
        }
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to process learner optical flow: {exc}",
        ) from exc
    finally:
        await file.close()


async def _save_uploaded_video(file: UploadFile, destination: Path) -> None:
    if not file.filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="A learner video file is required.",
        )

    with destination.open("wb") as output:
        while True:
            chunk = await file.read(1024 * 1024)
            if not chunk:
                break
            output.write(chunk)
