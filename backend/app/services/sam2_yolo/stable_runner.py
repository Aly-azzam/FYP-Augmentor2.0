from __future__ import annotations

import argparse
import contextlib
import gc
import json
import logging
import math
import os
import shutil
import sys
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import quote

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import cv2
import numpy as np
import torch

from app.services.sam2_yolo.cleaned_trajectory import (
    build_cleaned_trajectory_artifact,
    smooth_trajectory,
    write_smoothed_cutting_path_overlay_video,
    write_trajectory_line_overlay_video,
)
from app.services.sam2_yolo.region_metrics import compute_region_metrics
from app.services.sam2_yolo.schemas import (
    CLEANED_TRAJECTORY_FILENAME,
    DEFAULT_TRAJECTORY_POINT_MODE,
    DEBUG_FIRST_FRAME_FILENAME,
    METRICS_FILENAME,
    MODEL_NAME,
    OVERLAY_FILENAME,
    PROMPT_SOURCE,
    RAW_FILENAME,
    SUMMARY_FILENAME,
    SMOOTHED_CUTTING_PATH_OVERLAY_FILENAME,
    SMOOTH_POLYLINE_OVERLAY_FILENAME,
    TRACKING_POINT_TYPE,
    TRACKING_QUALITY_NOTE,
    TRAJECTORY_SMOOTHED_FILENAME,
    TRAJECTORY_POINT_MODES,
    ExtractedVideo,
    InitialPrompt,
    build_frame_payload,
    mask_to_metrics,
    regions_from_frames,
    trajectory_from_frames,
)
from app.services.sam2_yolo.trajectory_metrics import compute_trajectory_metrics
from app.services.sam2_yolo.visualization import write_sam2_yolo_overlay_video
from app.services.sam2_yolo.yolo_scissors_detector import (
    DEFAULT_ROBOFLOW_CONFIDENCE,
    DEFAULT_YOLO_SCISSORS_OUTPUT_ROOT,
    YoloScissorsDetectionError,
    get_or_create_yolo_scissors_raw_artifact,
    prompt_from_yolo_scissors_artifact,
)


BACKEND_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUTPUT_ROOT = BACKEND_ROOT / "storage" / "outputs" / "sam2_yolo" / "runs"
DEFAULT_FRAME_STRIDE = 5
DEFAULT_MAX_PROCESSED_FRAMES: int | None = None
logger = logging.getLogger(__name__)

_CPU_STABILITY_CONFIGURED = False


@dataclass(slots=True)
class StableDeviceChoice:
    device: torch.device
    warning: str | None = None


@dataclass(slots=True)
class StableTrackingResult:
    frames: list[dict[str, Any]]
    masks_by_processed_frame: dict[int, np.ndarray]
    extracted_video: ExtractedVideo
    device: str
    device_warning: str | None


def run_stable_yolo_sam2_tracking(
    *,
    video_path: str | Path,
    output_dir: str | Path,
    stride: int = DEFAULT_FRAME_STRIDE,
    max_processed_frames: int | None = None,
    use_gpu: bool = True,
    tracking_point_type: str = TRACKING_POINT_TYPE,
    trajectory_point_mode: str = DEFAULT_TRAJECTORY_POINT_MODE,
    save_debug: bool = False,
    debug_gpu: bool = False,
    roboflow_confidence: float = DEFAULT_ROBOFLOW_CONFIDENCE,
    sam2_checkpoint: str | None = None,
    sam2_config: str | None = None,
    sam2_model_id: str | None = None,
) -> dict[str, Any]:
    if tracking_point_type != TRACKING_POINT_TYPE:
        raise ValueError("Only tracking_point_type='bbox_center' is supported for now")
    if trajectory_point_mode not in TRAJECTORY_POINT_MODES:
        raise ValueError(f"trajectory_point_mode must be one of {TRAJECTORY_POINT_MODES}")

    run_dir = Path(output_dir).expanduser().resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    run_id = run_dir.name
    video = Path(video_path).expanduser().resolve()
    if not video.is_file():
        raise FileNotFoundError(f"video_path does not exist: {video}")

    raw_path = run_dir / RAW_FILENAME
    metrics_path = run_dir / METRICS_FILENAME
    summary_path = run_dir / SUMMARY_FILENAME
    overlay_path = run_dir / OVERLAY_FILENAME
    cleaned_trajectory_path = run_dir / CLEANED_TRAJECTORY_FILENAME
    trajectory_smoothed_path = run_dir / TRAJECTORY_SMOOTHED_FILENAME
    smooth_polyline_overlay_path = run_dir / SMOOTH_POLYLINE_OVERLAY_FILENAME
    smoothed_cutting_path_overlay_path = run_dir / SMOOTHED_CUTTING_PATH_OVERLAY_FILENAME
    debug_path = run_dir / DEBUG_FIRST_FRAME_FILENAME if save_debug else None
    frame_cap = _resolved_frame_cap(max_processed_frames)

    device_choice = choose_stable_device(use_gpu=use_gpu)
    initial_prompt: InitialPrompt | None = None

    try:
        yolo_payload, _, _ = get_or_create_yolo_scissors_raw_artifact(
            video_path=video,
            run_id=run_id,
            output_root=DEFAULT_YOLO_SCISSORS_OUTPUT_ROOT,
            confidence_threshold=roboflow_confidence,
            frame_stride=stride,
            max_scanned_frames=30,
        )
        logger.info("[SAM2] using shared YOLO best_prompt_frame=%s", yolo_payload.get("best_prompt_frame"))
        initial_prompt = prompt_from_yolo_scissors_artifact(
            payload=yolo_payload,
            video_path=video,
            debug_image_path=debug_path,
        )
        logger.info(
            "[SAM2] shrunken bbox=%s",
            initial_prompt.sam2_prompt_bbox or initial_prompt.bbox,
        )
    except YoloScissorsDetectionError as exc:
        summary = _build_failed_summary(
            run_id=run_id,
            video_path=video,
            run_dir=run_dir,
            raw_path=raw_path,
            metrics_path=metrics_path,
            summary_path=summary_path,
            overlay_path=overlay_path,
            frame_stride=stride,
            max_processed_frames=frame_cap,
            stage="yolo_prompt",
            failure_reason=str(exc),
            device_attempted=device_choice.device.type,
            fallback_attempted=False,
            device_warning=device_choice.warning,
        )
        _write_json(summary_path, summary)
        return summary

    try:
        tracking_result = _run_stable_sam2_tracking(
            video_path=video,
            initial_prompt=initial_prompt,
            frame_stride=stride,
            run_dir=run_dir,
            device_choice=device_choice,
            max_processed_frames=frame_cap,
            sam2_checkpoint=sam2_checkpoint,
            sam2_config=sam2_config,
            sam2_model_id=sam2_model_id,
            debug_gpu=debug_gpu,
        )
    except RuntimeError as exc:
        if device_choice.device.type == "cuda" and _is_cuda_oom(exc):
            torch.cuda.empty_cache()
            cpu_choice = StableDeviceChoice(
                device=torch.device("cpu"),
                warning="CUDA out of memory during SAM2, retried on CPU for SAM2",
            )
            try:
                tracking_result = _run_stable_sam2_tracking(
                    video_path=video,
                    initial_prompt=initial_prompt,
                    frame_stride=stride,
                    run_dir=run_dir,
                    device_choice=cpu_choice,
                    max_processed_frames=frame_cap,
                    sam2_checkpoint=sam2_checkpoint,
                    sam2_config=sam2_config,
                    sam2_model_id=sam2_model_id,
                    debug_gpu=debug_gpu,
                )
            except Exception as fallback_exc:  # noqa: BLE001
                summary = _sam2_failed_summary(
                    exc=fallback_exc,
                    run_id=run_id,
                    video_path=video,
                    run_dir=run_dir,
                    raw_path=raw_path,
                    metrics_path=metrics_path,
                    summary_path=summary_path,
                    overlay_path=overlay_path,
                    frame_stride=stride,
                    max_processed_frames=frame_cap,
                    stage="sam2_propagation",
                    device_attempted="cuda",
                    fallback_attempted=True,
                    device_warning=cpu_choice.warning,
                )
                _write_json(summary_path, summary)
                return summary
        else:
            summary = _sam2_failed_summary(
                exc=exc,
                run_id=run_id,
                video_path=video,
                run_dir=run_dir,
                raw_path=raw_path,
                metrics_path=metrics_path,
                summary_path=summary_path,
                overlay_path=overlay_path,
                frame_stride=stride,
                max_processed_frames=frame_cap,
                stage="sam2_propagation",
                device_attempted=device_choice.device.type,
                fallback_attempted=False,
                device_warning=device_choice.warning,
            )
            _write_json(summary_path, summary)
            return summary
    except Exception as exc:  # noqa: BLE001
        summary = _sam2_failed_summary(
            exc=exc,
            run_id=run_id,
            video_path=video,
            run_dir=run_dir,
            raw_path=raw_path,
            metrics_path=metrics_path,
            summary_path=summary_path,
            overlay_path=overlay_path,
            frame_stride=stride,
            max_processed_frames=frame_cap,
            stage="sam2_initialization",
            device_attempted=device_choice.device.type,
            fallback_attempted=False,
            device_warning=device_choice.warning,
        )
        _write_json(summary_path, summary)
        return summary

    raw_payload = _build_raw_payload(
        run_id=run_id,
        video_path=video,
        device=tracking_result.device,
        frame_stride=stride,
        max_processed_frames=frame_cap,
        original_frame_count=tracking_result.extracted_video.source_frame_count,
        initial_prompt=initial_prompt,
        frames=tracking_result.frames,
    )
    if yolo_payload.get("yolo_backend"):
        raw_payload["yolo_backend"] = yolo_payload.get("yolo_backend")
    if yolo_payload.get("yolo_weights_path"):
        raw_payload["yolo_weights_path"] = yolo_payload.get("yolo_weights_path")
    metrics_payload = _build_metrics_payload(raw_payload)
    summary_payload = _build_success_summary(
        raw_payload=raw_payload,
        metrics_payload=metrics_payload,
        run_dir=run_dir,
        raw_path=raw_path,
        metrics_path=metrics_path,
        summary_path=summary_path,
        overlay_path=overlay_path,
        device_warning=tracking_result.device_warning,
    )

    _write_json(raw_path, raw_payload)
    _write_json(metrics_path, metrics_payload)

    try:
        write_sam2_yolo_overlay_video(
            frame_dir=tracking_result.extracted_video.frame_dir,
            frames=tracking_result.frames,
            masks_by_processed_frame=tracking_result.masks_by_processed_frame,
            output_path=overlay_path,
            fps=tracking_result.extracted_video.output_fps,
            trajectory_metrics=metrics_payload["trajectory_metrics"],
            trajectory_point_mode=trajectory_point_mode,
        )
    except Exception as exc:  # noqa: BLE001
        summary = _build_failed_summary(
            run_id=run_id,
            video_path=video,
            run_dir=run_dir,
            raw_path=raw_path,
            metrics_path=metrics_path,
            summary_path=summary_path,
            overlay_path=overlay_path,
            frame_stride=stride,
            max_processed_frames=frame_cap,
            stage="visualization",
            failure_reason=str(exc),
            device_attempted=raw_payload["device"],
            fallback_attempted=False,
            device_warning=tracking_result.device_warning,
        )
        _write_json(summary_path, summary)
        return summary

    try:
        cleaned_trajectory_payload = build_cleaned_trajectory_artifact(
            raw_json_path=raw_path,
            metrics_json_path=metrics_path,
            output_path=cleaned_trajectory_path,
            trajectory_point_mode=trajectory_point_mode,
        )
        write_trajectory_line_overlay_video(
            frame_dir=tracking_result.extracted_video.frame_dir,
            raw_frames=tracking_result.frames,
            cleaned_trajectory=cleaned_trajectory_payload,
            output_path=smooth_polyline_overlay_path,
            fps=tracking_result.extracted_video.output_fps,
        )
        smoothed_trajectory_payload = smooth_trajectory(
            cleaned_json_path=str(cleaned_trajectory_path),
            output_path=str(trajectory_smoothed_path),
        )
        write_smoothed_cutting_path_overlay_video(
            frame_dir=tracking_result.extracted_video.frame_dir,
            raw_frames=tracking_result.frames,
            smoothed_trajectory=smoothed_trajectory_payload,
            output_path=smoothed_cutting_path_overlay_path,
            fps=tracking_result.extracted_video.output_fps,
        )
        summary_payload.update(
            {
                "cleaned_trajectory_json_path": str(cleaned_trajectory_path),
                "smooth_polyline_overlay_video_path": str(smooth_polyline_overlay_path),
                "smooth_polyline_overlay_video_url": _storage_url_for(smooth_polyline_overlay_path),
                "cleaned_trajectory_usable_for_corridor": bool(
                    cleaned_trajectory_payload.get("usable_for_corridor", False)
                ),
                "trajectory_smoothed_json_path": str(trajectory_smoothed_path),
                "smoothed_cutting_path_overlay_video_path": str(smoothed_cutting_path_overlay_path),
                "smoothed_cutting_path_overlay_video_url": _storage_url_for(smoothed_cutting_path_overlay_path),
                "smoothed_trajectory_usable_for_corridor": bool(
                    smoothed_trajectory_payload.get("usable_for_corridor", False)
                ),
            }
        )
    except Exception as exc:  # noqa: BLE001 - keep existing SAM2+YOLO outputs usable.
        logger.exception("[SAM2] cleaned trajectory post-processing failed")
        summary_payload["cleaned_trajectory_error"] = str(exc)

    _write_json(summary_path, summary_payload)
    return summary_payload


def run_stable_yolo_sam2_run(
    *,
    video_path: str | Path,
    output_root: str | Path = DEFAULT_OUTPUT_ROOT,
    run_id: str | None = None,
    stride: int = DEFAULT_FRAME_STRIDE,
    max_processed_frames: int | None = None,
    use_gpu: bool = True,
    tracking_point_type: str = TRACKING_POINT_TYPE,
    trajectory_point_mode: str = DEFAULT_TRAJECTORY_POINT_MODE,
    save_debug: bool = False,
    debug_gpu: bool = False,
    roboflow_confidence: float = DEFAULT_ROBOFLOW_CONFIDENCE,
    sam2_checkpoint: str | None = None,
    sam2_config: str | None = None,
    sam2_model_id: str | None = None,
) -> dict[str, Any]:
    resolved_run_id = run_id or str(uuid.uuid4())
    output_dir = Path(output_root).expanduser().resolve() / resolved_run_id
    if output_dir.exists():
        raise FileExistsError(f"SAM2+YOLO output already exists: {output_dir}")
    return run_stable_yolo_sam2_tracking(
        video_path=video_path,
        output_dir=output_dir,
        stride=stride,
        max_processed_frames=max_processed_frames,
        use_gpu=use_gpu,
        tracking_point_type=tracking_point_type,
        trajectory_point_mode=trajectory_point_mode,
        save_debug=save_debug,
        debug_gpu=debug_gpu,
        roboflow_confidence=roboflow_confidence,
        sam2_checkpoint=sam2_checkpoint,
        sam2_config=sam2_config,
        sam2_model_id=sam2_model_id,
    )


def choose_stable_device(use_gpu: bool = True) -> StableDeviceChoice:
    if not use_gpu or not torch.cuda.is_available():
        _configure_cpu_stability()
        warning = "CUDA unavailable, SAM2 ran on CPU" if use_gpu else "SAM2 GPU disabled, ran on CPU"
        return StableDeviceChoice(device=torch.device("cpu"), warning=warning)

    return StableDeviceChoice(device=torch.device("cuda"), warning=None)


def _run_stable_sam2_tracking(
    *,
    video_path: Path,
    initial_prompt: InitialPrompt,
    frame_stride: int,
    run_dir: Path,
    device_choice: StableDeviceChoice,
    max_processed_frames: int | None,
    sam2_checkpoint: str | None,
    sam2_config: str | None,
    sam2_model_id: str | None,
    debug_gpu: bool,
) -> StableTrackingResult:
    device = device_choice.device
    if device.type == "cpu":
        _configure_cpu_stability()

    predictor: Any | None = None
    inference_state: dict[str, Any] | None = None
    model_id_for_log = sam2_model_id or os.getenv("SAM2_MODEL_ID", "facebook/sam2.1-hiera-tiny")
    extracted = extract_strided_frames_stable(
        video_path=video_path,
        frame_dir=run_dir / "frames",
        frame_stride=frame_stride,
        required_frame_index=initial_prompt.frame_index,
        max_processed_frames=max_processed_frames,
    )
    if debug_gpu:
        _print_gpu_diagnostics_header(
            device=device,
            extracted=extracted,
            model_id=model_id_for_log,
        )
    log_cuda_memory("before build predictor", enabled=debug_gpu)

    frame_results: list[dict[str, Any]] = []
    masks_by_processed_frame: dict[int, np.ndarray] = {}

    try:
        predictor = build_video_predictor_stable(
            device=device,
            sam2_checkpoint=sam2_checkpoint,
            sam2_config=sam2_config,
            sam2_model_id=sam2_model_id,
        )
        log_cuda_memory("after build predictor", enabled=debug_gpu)

        points = np.array([initial_prompt.point], dtype=np.float32)
        labels = np.array([1], dtype=np.int32)
        box = np.array(initial_prompt.sam2_prompt_bbox or initial_prompt.bbox, dtype=np.float32)
        prompt_processed_frame = _processed_frame_for_original(
            extracted.processed_to_original,
            initial_prompt.frame_index,
        )

        with torch.inference_mode(), autocast_context_stable(device):
            log_cuda_memory("before init_state", enabled=debug_gpu)
            inference_state = predictor.init_state(
                video_path=str(extracted.frame_dir),
                async_loading_frames=False,
            )
            log_cuda_memory("after init_state", enabled=debug_gpu)
            log_cuda_memory("before add_new_points_or_box", enabled=debug_gpu)
            predictor.add_new_points_or_box(
                inference_state=inference_state,
                frame_idx=prompt_processed_frame,
                obj_id=1,
                points=points,
                labels=labels,
                box=box,
                normalize_coords=True,
            )
            log_cuda_memory("after add_new_points_or_box", enabled=debug_gpu)

            for propagation_count, (
                processed_frame_idx,
                object_ids,
                mask_logits,
            ) in enumerate(predictor.propagate_in_video(inference_state), start=1):
                if propagation_count == 1 or propagation_count % 10 == 0:
                    log_cuda_memory(
                        f"during propagation frame {propagation_count}",
                        enabled=debug_gpu,
                    )
                processed_idx = int(processed_frame_idx)
                mask = _mask_for_object(object_ids, mask_logits, obj_id=1)
                mask_area, mask_centroid, mask_bbox, mask_extreme_points = mask_to_metrics(mask)
                original_frame_idx = extracted.processed_to_original[processed_idx]
                timestamp_sec = extracted.processed_to_timestamp[processed_idx]
                masks_by_processed_frame[processed_idx] = mask
                frame_results.append(
                    build_frame_payload(
                        frame_index=original_frame_idx,
                        processed_frame_index=processed_idx,
                        timestamp_sec=timestamp_sec,
                        mask_bbox=mask_bbox,
                        mask_centroid=mask_centroid,
                        mask_area=mask_area,
                        mask_extreme_points=mask_extreme_points,
                    )
                )
    finally:
        cleanup_sam2_runtime(
            predictor=predictor,
            inference_state=inference_state,
            debug_gpu=debug_gpu,
        )

    result = StableTrackingResult(
        frames=frame_results,
        masks_by_processed_frame=masks_by_processed_frame,
        extracted_video=extracted,
        device=device.type,
        device_warning=device_choice.warning,
    )
    return result


def extract_strided_frames_stable(
    *,
    video_path: Path,
    frame_dir: Path,
    frame_stride: int,
    required_frame_index: int,
    max_processed_frames: int | None,
) -> ExtractedVideo:
    if frame_stride < 1:
        raise ValueError("frame_stride must be >= 1")

    if frame_dir.exists():
        shutil.rmtree(frame_dir)
    frame_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    source_fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    original_frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if width <= 0 or height <= 0:
        cap.release()
        raise RuntimeError(f"Invalid video dimensions for {video_path}")

    processed_to_original: dict[int, int] = {}
    processed_to_timestamp: dict[int, float] = {}
    frame_index = 0
    processed_index = 0
    prompt_included = False
    cap_limit = max_processed_frames if max_processed_frames and max_processed_frames > 0 else None

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            is_required = frame_index == required_frame_index
            should_keep = frame_index % frame_stride == 0 or is_required
            if should_keep:
                if cap_limit is None or processed_index < cap_limit:
                    write_index = processed_index
                    processed_index += 1
                elif is_required and not prompt_included:
                    write_index = cap_limit - 1
                    stale_path = frame_dir / f"{write_index:06d}.jpg"
                    if stale_path.exists():
                        stale_path.unlink()
                else:
                    frame_index += 1
                    continue

                frame_path = frame_dir / f"{write_index:06d}.jpg"
                if not cv2.imwrite(str(frame_path), frame):
                    raise RuntimeError(f"Could not write extracted frame: {frame_path}")
                processed_to_original[write_index] = frame_index
                processed_to_timestamp[write_index] = round(frame_index / max(source_fps, 1e-6), 6)
                prompt_included = prompt_included or is_required

                if cap_limit is not None and processed_index >= cap_limit and prompt_included:
                    frame_index += 1
                    break

            frame_index += 1
    finally:
        cap.release()

    if not processed_to_original:
        raise RuntimeError(f"No frames were extracted from {video_path}")
    if required_frame_index not in set(processed_to_original.values()):
        raise RuntimeError(
            f"Prompt frame {required_frame_index} was not included in extracted frames"
        )

    return ExtractedVideo(
        frame_dir=frame_dir,
        processed_to_original=dict(sorted(processed_to_original.items())),
        processed_to_timestamp=dict(sorted(processed_to_timestamp.items())),
        source_fps=source_fps,
        output_fps=max(source_fps / frame_stride, 1.0),
        width=width,
        height=height,
        source_frame_count=original_frame_count or frame_index,
    )


def build_video_predictor_stable(
    *,
    device: torch.device,
    sam2_checkpoint: str | None = None,
    sam2_config: str | None = None,
    sam2_model_id: str | None = None,
) -> Any:
    _ensure_local_sam2_on_path()
    from sam2.build_sam import (  # noqa: PLC0415
        build_sam2_video_predictor,
        build_sam2_video_predictor_hf,
    )

    local_checkpoint = BACKEND_ROOT / "models" / "sam2" / "sam2_hiera_tiny.pt"
    local_config = "configs/sam2/sam2_hiera_t.yaml"
    if sam2_checkpoint is None and sam2_config is None and local_checkpoint.is_file():
        sam2_checkpoint = str(local_checkpoint)
        sam2_config = local_config

    if sam2_checkpoint and sam2_config:
        return build_sam2_video_predictor(
            sam2_config,
            sam2_checkpoint,
            device=str(device),
        )

    model_id = sam2_model_id or os.getenv("SAM2_MODEL_ID", "facebook/sam2.1-hiera-tiny")
    return build_sam2_video_predictor_hf(model_id, device=str(device))


def autocast_context_stable(device: torch.device) -> contextlib.AbstractContextManager:
    if device.type == "cuda":
        return torch.autocast("cuda", dtype=torch.bfloat16)
    return contextlib.nullcontext()


def log_cuda_memory(stage: str, *, enabled: bool = True) -> None:
    if not enabled or not torch.cuda.is_available():
        return
    print(f"[CUDA] {stage}", flush=True)
    print("allocated MB:", torch.cuda.memory_allocated() / 1024**2, flush=True)
    print("reserved MB:", torch.cuda.memory_reserved() / 1024**2, flush=True)
    print("max allocated MB:", torch.cuda.max_memory_allocated() / 1024**2, flush=True)


def cleanup_sam2_runtime(
    *,
    predictor: Any | None,
    inference_state: dict[str, Any] | None,
    debug_gpu: bool = False,
) -> None:
    log_cuda_memory("before cleanup", enabled=debug_gpu)
    try:
        del inference_state
    except Exception:
        pass
    try:
        del predictor
    except Exception:
        pass
    gc.collect()
    if torch.cuda.is_available():
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass
        try:
            torch.cuda.ipc_collect()
        except Exception:
            pass
        try:
            torch.cuda.reset_peak_memory_stats()
        except Exception:
            pass
    log_cuda_memory("after cleanup", enabled=debug_gpu)


def _print_gpu_diagnostics_header(
    *,
    device: torch.device,
    extracted: ExtractedVideo,
    model_id: str,
) -> None:
    print("[SAM2-YOLO] GPU diagnostics", flush=True)
    print("torch version:", torch.__version__, flush=True)
    print("torch cuda version:", torch.version.cuda, flush=True)
    print("cuda available:", torch.cuda.is_available(), flush=True)
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        print("gpu name:", props.name, flush=True)
        print("total VRAM MB:", props.total_memory / 1024**2, flush=True)
    print("selected device:", device.type, flush=True)
    print("extracted frame count:", len(extracted.processed_to_original), flush=True)
    print("frame resolution:", f"{extracted.width}x{extracted.height}", flush=True)
    print("SAM2 model id:", model_id, flush=True)
    print("autocast enabled:", device.type == "cuda", flush=True)
    print("async_loading_frames:", False, flush=True)


def _configure_cpu_stability() -> None:
    global _CPU_STABILITY_CONFIGURED
    if _CPU_STABILITY_CONFIGURED:
        return
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    try:
        torch.set_num_threads(1)
    except RuntimeError:
        pass
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    torch.backends.mkldnn.enabled = False
    _CPU_STABILITY_CONFIGURED = True


def _mask_for_object(object_ids: list[int], mask_logits: torch.Tensor, obj_id: int) -> np.ndarray:
    object_id_list = [int(object_id) for object_id in object_ids]
    if obj_id not in object_id_list:
        height, width = mask_logits.shape[-2:]
        return np.zeros((height, width), dtype=bool)

    object_index = object_id_list.index(obj_id)
    mask = (mask_logits[object_index] > 0.0).detach().cpu().numpy()
    if mask.ndim == 3:
        mask = mask[0]
    return mask.astype(bool)


def _processed_frame_for_original(
    processed_to_original: dict[int, int],
    original_frame_index: int,
) -> int:
    for processed_frame_index, mapped_original in processed_to_original.items():
        if mapped_original == original_frame_index:
            return processed_frame_index
    raise RuntimeError(
        f"Prompt frame {original_frame_index} was not included in the extracted stride frames"
    )


def _ensure_local_sam2_on_path() -> None:
    local_sam2 = BACKEND_ROOT / "vendor" / "sam2"
    if local_sam2.is_dir():
        local_sam2_str = str(local_sam2)
        if local_sam2_str not in sys.path:
            sys.path.insert(0, local_sam2_str)


def _resolved_frame_cap(max_processed_frames: int | None) -> int | None:
    if max_processed_frames is not None:
        return max_processed_frames if max_processed_frames > 0 else None
    return None


def _is_cuda_oom(exc: RuntimeError) -> bool:
    message = str(exc).lower()
    return "out of memory" in message or "cuda oom" in message


def _build_raw_payload(
    *,
    run_id: str,
    video_path: Path,
    device: str,
    frame_stride: int,
    max_processed_frames: int | None,
    original_frame_count: int,
    initial_prompt: InitialPrompt,
    frames: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "run_id": run_id,
        "video_path": str(video_path),
        "model": MODEL_NAME,
        "device": device,
        "frame_stride": frame_stride,
        "original_frame_count": original_frame_count,
        "expected_strided_frames": _expected_strided_frames(original_frame_count, frame_stride),
        "max_processed_frames": max_processed_frames,
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
        "original_frame_count": raw_payload["original_frame_count"],
        "expected_strided_frames": raw_payload["expected_strided_frames"],
        "max_processed_frames": raw_payload["max_processed_frames"],
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
        "original_frame_count": raw_payload["original_frame_count"],
        "expected_strided_frames": raw_payload["expected_strided_frames"],
        "max_processed_frames": raw_payload["max_processed_frames"],
        "video_path": raw_payload["video_path"],
        "raw_json_path": str(raw_path),
        "metrics_json_path": str(metrics_path),
        "summary_json_path": str(summary_path),
        "overlay_video_path": str(overlay_path),
        "overlay_video_url": _storage_url_for(overlay_path),
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


def _sam2_failed_summary(
    *,
    exc: Exception,
    run_id: str,
    video_path: Path,
    run_dir: Path,
    raw_path: Path,
    metrics_path: Path,
    summary_path: Path,
    overlay_path: Path,
    frame_stride: int,
    max_processed_frames: int | None,
    stage: str,
    device_attempted: str,
    fallback_attempted: bool,
    device_warning: str | None,
) -> dict[str, Any]:
    return _build_failed_summary(
        run_id=run_id,
        video_path=video_path,
        run_dir=run_dir,
        raw_path=raw_path,
        metrics_path=metrics_path,
        summary_path=summary_path,
        overlay_path=overlay_path,
        frame_stride=frame_stride,
        max_processed_frames=max_processed_frames,
        stage=stage,
        failure_reason=str(exc),
        device_attempted=device_attempted,
        fallback_attempted=fallback_attempted,
        device_warning=device_warning,
    )


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
    max_processed_frames: int | None,
    stage: str,
    failure_reason: str,
    device_attempted: str,
    fallback_attempted: bool,
    device_warning: str | None,
) -> dict[str, Any]:
    summary = {
        "run_id": run_id,
        "status": "failed",
        "stage": stage,
        "failure_reason": failure_reason,
        "device_attempted": device_attempted,
        "fallback_attempted": fallback_attempted,
        "model": MODEL_NAME,
        "device": device_attempted,
        "frame_stride": frame_stride,
        "original_frame_count": None,
        "expected_strided_frames": None,
        "max_processed_frames": max_processed_frames,
        "video_path": str(video_path),
        "raw_json_path": str(raw_path),
        "metrics_json_path": str(metrics_path),
        "summary_json_path": str(summary_path),
        "overlay_video_path": str(overlay_path),
        "overlay_video_url": _storage_url_for(overlay_path),
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


def _storage_url_for(path: Path) -> str | None:
    storage_root = BACKEND_ROOT / "storage"
    try:
        storage_key = path.resolve().relative_to(storage_root.resolve()).as_posix()
    except ValueError:
        return None
    return f"/storage/{quote(storage_key, safe='/')}"


def _expected_strided_frames(original_frame_count: int, frame_stride: int) -> int | None:
    if original_frame_count <= 0:
        return None
    return int(math.ceil(original_frame_count / max(frame_stride, 1)))


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Stable YOLO+SAM2 scissors tracking using the yolo_scissors_test SAM2 flow."
    )
    parser.add_argument("--video_path", required=True, help="Path to the input video.")
    parser.add_argument("--stride", type=int, default=DEFAULT_FRAME_STRIDE, help="SAM2 frame stride.")
    parser.add_argument(
        "--max_processed_frames",
        type=int,
        default=0,
        help="Maximum processed JPEG frames SAM2 should load. Use 0 for no cap.",
    )
    parser.add_argument("--output_dir", default=None, help="Exact output run directory.")
    parser.add_argument("--output_root", default=str(DEFAULT_OUTPUT_ROOT), help="Runs output root.")
    parser.add_argument("--run_id", default=None, help="Optional run id. Defaults to a UUID.")
    parser.add_argument(
        "--use_gpu",
        action="store_true",
        default=True,
        help="Use CUDA when safe. Enabled by default.",
    )
    parser.add_argument("--no-gpu", "--no_gpu", action="store_false", dest="use_gpu")
    parser.add_argument(
        "--tracking_point_type",
        choices=[TRACKING_POINT_TYPE],
        default=TRACKING_POINT_TYPE,
    )
    parser.add_argument(
        "--trajectory_point_mode",
        choices=list(TRAJECTORY_POINT_MODES),
        default=DEFAULT_TRAJECTORY_POINT_MODE,
        help="Point source used for cleaned trajectory output and debug trajectory line.",
    )
    parser.add_argument("--save_debug", action="store_true", help="Save debug_first_frame.jpg.")
    parser.add_argument(
        "--roboflow_confidence",
        type=float,
        default=DEFAULT_ROBOFLOW_CONFIDENCE,
        help="Roboflow confidence threshold.",
    )
    parser.add_argument("--sam2_checkpoint", default=None, help="Optional local SAM2 checkpoint path.")
    parser.add_argument("--sam2_config", default=None, help="Optional local SAM2 config path/name.")
    parser.add_argument("--sam2_model_id", default=None, help="Optional Hugging Face SAM2 model id.")
    parser.add_argument("--debug_gpu", action="store_true", help="Print CUDA memory diagnostics.")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    frame_cap = args.max_processed_frames if args.max_processed_frames > 0 else None
    if args.output_dir:
        summary = run_stable_yolo_sam2_tracking(
            video_path=args.video_path,
            output_dir=args.output_dir,
            stride=args.stride,
            max_processed_frames=frame_cap,
            use_gpu=args.use_gpu,
            tracking_point_type=args.tracking_point_type,
            trajectory_point_mode=args.trajectory_point_mode,
            save_debug=args.save_debug,
            debug_gpu=args.debug_gpu,
            roboflow_confidence=args.roboflow_confidence,
            sam2_checkpoint=args.sam2_checkpoint,
            sam2_config=args.sam2_config,
            sam2_model_id=args.sam2_model_id,
        )
    else:
        summary = run_stable_yolo_sam2_run(
            video_path=args.video_path,
            output_root=args.output_root,
            run_id=args.run_id,
            stride=args.stride,
            max_processed_frames=frame_cap,
            use_gpu=args.use_gpu,
            tracking_point_type=args.tracking_point_type,
            trajectory_point_mode=args.trajectory_point_mode,
            save_debug=args.save_debug,
            debug_gpu=args.debug_gpu,
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


# ── New entry point — accepts a pre-built InitialPrompt, skips YOLO ──────────
# The original run_stable_yolo_sam2_tracking() is untouched above.

def run_stable_yolo_sam2_tracking_with_prompt(
    *,
    video_path: str | Path,
    output_dir: str | Path,
    initial_prompt: InitialPrompt,
    stride: int = DEFAULT_FRAME_STRIDE,
    max_processed_frames: int | None = None,
    use_gpu: bool = True,
    trajectory_point_mode: str = DEFAULT_TRAJECTORY_POINT_MODE,
    save_debug: bool = False,
    debug_gpu: bool = False,
    sam2_checkpoint: str | None = None,
    sam2_config: str | None = None,
    sam2_model_id: str | None = None,
) -> dict[str, Any]:
    """Identical to run_stable_yolo_sam2_tracking but skips the YOLO step.

    Uses the provided *initial_prompt* directly instead of running YOLO detection.
    Called by the evaluation orchestrator which runs YOLO once via run_shared_yolo
    and feeds the result to both SAM2 and the angle pipeline.
    """
    if trajectory_point_mode not in TRAJECTORY_POINT_MODES:
        raise ValueError(f"trajectory_point_mode must be one of {TRAJECTORY_POINT_MODES}")

    run_dir = Path(output_dir).expanduser().resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    run_id = run_dir.name
    video = Path(video_path).expanduser().resolve()
    if not video.is_file():
        raise FileNotFoundError(f"video_path does not exist: {video}")

    raw_path = run_dir / RAW_FILENAME
    metrics_path = run_dir / METRICS_FILENAME
    summary_path = run_dir / SUMMARY_FILENAME
    overlay_path = run_dir / OVERLAY_FILENAME
    cleaned_trajectory_path = run_dir / CLEANED_TRAJECTORY_FILENAME
    trajectory_smoothed_path = run_dir / TRAJECTORY_SMOOTHED_FILENAME
    smooth_polyline_overlay_path = run_dir / SMOOTH_POLYLINE_OVERLAY_FILENAME
    smoothed_cutting_path_overlay_path = run_dir / SMOOTHED_CUTTING_PATH_OVERLAY_FILENAME
    frame_cap = _resolved_frame_cap(max_processed_frames)

    device_choice = choose_stable_device(use_gpu=use_gpu)

    # YOLO step is skipped — prompt provided by caller.
    yolo_payload: dict[str, Any] = {
        "yolo_backend": "shared_yolo_runner",
        "yolo_weights_path": str(BACKEND_ROOT / "models" / "yolo" / "best.pt"),
    }

    try:
        tracking_result = _run_stable_sam2_tracking(
            video_path=video,
            initial_prompt=initial_prompt,
            frame_stride=stride,
            run_dir=run_dir,
            device_choice=device_choice,
            max_processed_frames=frame_cap,
            sam2_checkpoint=sam2_checkpoint,
            sam2_config=sam2_config,
            sam2_model_id=sam2_model_id,
            debug_gpu=debug_gpu,
        )
    except RuntimeError as exc:
        if device_choice.device.type == "cuda" and _is_cuda_oom(exc):
            torch.cuda.empty_cache()
            cpu_choice = StableDeviceChoice(
                device=torch.device("cpu"),
                warning="CUDA out of memory during SAM2, retried on CPU for SAM2",
            )
            try:
                tracking_result = _run_stable_sam2_tracking(
                    video_path=video,
                    initial_prompt=initial_prompt,
                    frame_stride=stride,
                    run_dir=run_dir,
                    device_choice=cpu_choice,
                    max_processed_frames=frame_cap,
                    sam2_checkpoint=sam2_checkpoint,
                    sam2_config=sam2_config,
                    sam2_model_id=sam2_model_id,
                    debug_gpu=debug_gpu,
                )
            except Exception as fallback_exc:  # noqa: BLE001
                summary = _sam2_failed_summary(
                    exc=fallback_exc,
                    run_id=run_id,
                    video_path=video,
                    run_dir=run_dir,
                    raw_path=raw_path,
                    metrics_path=metrics_path,
                    summary_path=summary_path,
                    overlay_path=overlay_path,
                    frame_stride=stride,
                    max_processed_frames=frame_cap,
                    stage="sam2_propagation",
                    device_attempted="cuda",
                    fallback_attempted=True,
                    device_warning=cpu_choice.warning,
                )
                _write_json(summary_path, summary)
                return summary
        else:
            summary = _sam2_failed_summary(
                exc=exc,
                run_id=run_id,
                video_path=video,
                run_dir=run_dir,
                raw_path=raw_path,
                metrics_path=metrics_path,
                summary_path=summary_path,
                overlay_path=overlay_path,
                frame_stride=stride,
                max_processed_frames=frame_cap,
                stage="sam2_propagation",
                device_attempted=device_choice.device.type,
                fallback_attempted=False,
                device_warning=device_choice.warning,
            )
            _write_json(summary_path, summary)
            return summary
    except Exception as exc:  # noqa: BLE001
        summary = _sam2_failed_summary(
            exc=exc,
            run_id=run_id,
            video_path=video,
            run_dir=run_dir,
            raw_path=raw_path,
            metrics_path=metrics_path,
            summary_path=summary_path,
            overlay_path=overlay_path,
            frame_stride=stride,
            max_processed_frames=frame_cap,
            stage="sam2_initialization",
            device_attempted=device_choice.device.type,
            fallback_attempted=False,
            device_warning=device_choice.warning,
        )
        _write_json(summary_path, summary)
        return summary

    raw_payload = _build_raw_payload(
        run_id=run_id,
        video_path=video,
        device=tracking_result.device,
        frame_stride=stride,
        max_processed_frames=frame_cap,
        original_frame_count=tracking_result.extracted_video.source_frame_count,
        initial_prompt=initial_prompt,
        frames=tracking_result.frames,
    )
    if yolo_payload.get("yolo_backend"):
        raw_payload["yolo_backend"] = yolo_payload["yolo_backend"]
    if yolo_payload.get("yolo_weights_path"):
        raw_payload["yolo_weights_path"] = yolo_payload["yolo_weights_path"]

    metrics_payload = _build_metrics_payload(raw_payload)
    summary_payload = _build_success_summary(
        raw_payload=raw_payload,
        metrics_payload=metrics_payload,
        run_dir=run_dir,
        raw_path=raw_path,
        metrics_path=metrics_path,
        summary_path=summary_path,
        overlay_path=overlay_path,
        device_warning=tracking_result.device_warning,
    )

    _write_json(raw_path, raw_payload)
    _write_json(metrics_path, metrics_payload)

    try:
        write_sam2_yolo_overlay_video(
            frame_dir=tracking_result.extracted_video.frame_dir,
            frames=tracking_result.frames,
            masks_by_processed_frame=tracking_result.masks_by_processed_frame,
            output_path=overlay_path,
            fps=tracking_result.extracted_video.output_fps,
            trajectory_metrics=metrics_payload["trajectory_metrics"],
            trajectory_point_mode=trajectory_point_mode,
        )
    except Exception as exc:  # noqa: BLE001
        summary = _build_failed_summary(
            run_id=run_id,
            video_path=video,
            run_dir=run_dir,
            raw_path=raw_path,
            metrics_path=metrics_path,
            summary_path=summary_path,
            overlay_path=overlay_path,
            frame_stride=stride,
            max_processed_frames=frame_cap,
            stage="visualization",
            failure_reason=str(exc),
            device_attempted=raw_payload["device"],
            fallback_attempted=False,
            device_warning=tracking_result.device_warning,
        )
        _write_json(summary_path, summary)
        return summary

    try:
        cleaned_trajectory_payload = build_cleaned_trajectory_artifact(
            raw_json_path=raw_path,
            metrics_json_path=metrics_path,
            output_path=cleaned_trajectory_path,
            trajectory_point_mode=trajectory_point_mode,
        )
        write_trajectory_line_overlay_video(
            frame_dir=tracking_result.extracted_video.frame_dir,
            raw_frames=tracking_result.frames,
            cleaned_trajectory=cleaned_trajectory_payload,
            output_path=smooth_polyline_overlay_path,
            fps=tracking_result.extracted_video.output_fps,
        )
        smoothed_trajectory_payload = smooth_trajectory(
            cleaned_json_path=str(cleaned_trajectory_path),
            output_path=str(trajectory_smoothed_path),
        )
        write_smoothed_cutting_path_overlay_video(
            frame_dir=tracking_result.extracted_video.frame_dir,
            raw_frames=tracking_result.frames,
            smoothed_trajectory=smoothed_trajectory_payload,
            output_path=smoothed_cutting_path_overlay_path,
            fps=tracking_result.extracted_video.output_fps,
        )
        summary_payload.update(
            {
                "cleaned_trajectory_json_path": str(cleaned_trajectory_path),
                "smooth_polyline_overlay_video_path": str(smooth_polyline_overlay_path),
                "smooth_polyline_overlay_video_url": _storage_url_for(smooth_polyline_overlay_path),
                "cleaned_trajectory_usable_for_corridor": bool(
                    cleaned_trajectory_payload.get("usable_for_corridor", False)
                ),
                "trajectory_smoothed_json_path": str(trajectory_smoothed_path),
                "smoothed_cutting_path_overlay_video_path": str(smoothed_cutting_path_overlay_path),
                "smoothed_cutting_path_overlay_video_url": _storage_url_for(
                    smoothed_cutting_path_overlay_path
                ),
                "smoothed_trajectory_usable_for_corridor": bool(
                    smoothed_trajectory_payload.get("usable_for_corridor", False)
                ),
            }
        )
    except Exception as exc:  # noqa: BLE001
        logger.exception("[SAM2] cleaned trajectory post-processing failed")
        summary_payload["cleaned_trajectory_error"] = str(exc)

    _write_json(summary_path, summary_payload)
    return summary_payload
