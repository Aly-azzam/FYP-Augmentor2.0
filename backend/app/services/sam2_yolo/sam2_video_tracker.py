from __future__ import annotations

import contextlib
import os
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch

from app.services.sam2_yolo.schemas import (
    ExtractedVideo,
    InitialPrompt,
    build_frame_payload,
    mask_to_metrics,
)


BACKEND_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_MAX_PROCESSED_FRAMES = int(os.getenv("SAM2_MAX_PROCESSED_FRAMES", "0"))


@dataclass(slots=True)
class Sam2TrackingResult:
    frames: list[dict[str, Any]]
    masks_by_processed_frame: dict[int, np.ndarray]
    extracted_video: ExtractedVideo


def choose_device(use_gpu: bool = True) -> torch.device:
    if use_gpu and torch.cuda.is_available():
        return torch.device("cuda")
    # SAM2's image encoder can hit native MKLDNN crashes on Windows CPU builds.
    # Disabling it keeps CPU fallback stable, at the cost of slower inference.
    torch.backends.mkldnn.enabled = False
    return torch.device("cpu")


def device_warning(device: torch.device) -> str | None:
    if device.type == "cuda":
        return None
    return "CUDA unavailable, SAM2 ran on CPU"


def autocast_context(device: torch.device) -> contextlib.AbstractContextManager:
    if device.type == "cuda":
        return torch.autocast("cuda", dtype=torch.bfloat16)
    return contextlib.nullcontext()


def run_sam2_video_tracking(
    *,
    video_path: Path,
    initial_prompt: InitialPrompt,
    frame_stride: int,
    run_dir: Path,
    device: torch.device,
    sam2_checkpoint: str | None = None,
    sam2_config: str | None = None,
    sam2_model_id: str | None = None,
) -> Sam2TrackingResult:
    extracted = extract_strided_frames(
        video_path=video_path,
        frame_dir=run_dir / "frames",
        frame_stride=frame_stride,
        required_frame_index=initial_prompt.frame_index,
        max_processed_frames=DEFAULT_MAX_PROCESSED_FRAMES if DEFAULT_MAX_PROCESSED_FRAMES > 0 else None,
    )
    predictor = build_video_predictor(
        device=device,
        sam2_checkpoint=sam2_checkpoint,
        sam2_config=sam2_config,
        sam2_model_id=sam2_model_id,
    )

    points = np.array([initial_prompt.point], dtype=np.float32)
    labels = np.array([1], dtype=np.int32)
    prompt_processed_frame = _processed_frame_for_original(
        extracted.processed_to_original,
        initial_prompt.frame_index,
    )

    frame_results: list[dict[str, Any]] = []
    masks_by_processed_frame: dict[int, np.ndarray] = {}

    with torch.inference_mode(), autocast_context(device):
        inference_state = predictor.init_state(
            video_path=str(extracted.frame_dir),
            async_loading_frames=False,
            offload_video_to_cpu=device.type == "cuda",
            offload_state_to_cpu=device.type == "cuda",
        )
        predictor.add_new_points_or_box(
            inference_state=inference_state,
            frame_idx=prompt_processed_frame,
            obj_id=1,
            points=points,
            labels=labels,
            normalize_coords=True,
        )

        for processed_frame_idx, object_ids, mask_logits in predictor.propagate_in_video(
            inference_state
        ):
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

    return Sam2TrackingResult(
        frames=frame_results,
        masks_by_processed_frame=masks_by_processed_frame,
        extracted_video=extracted,
    )


def extract_strided_frames(
    *,
    video_path: Path,
    frame_dir: Path,
    frame_stride: int,
    required_frame_index: int,
    max_processed_frames: int | None = None,
) -> ExtractedVideo:
    if frame_stride < 1:
        raise ValueError("frame_stride must be >= 1")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    source_fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if width <= 0 or height <= 0:
        cap.release()
        raise RuntimeError(f"Invalid video dimensions for {video_path}")

    if frame_dir.exists():
        shutil.rmtree(frame_dir)
    frame_dir.mkdir(parents=True, exist_ok=True)
    processed_to_original: dict[int, int] = {}
    processed_to_timestamp: dict[int, float] = {}
    frame_index = 0
    processed_index = 0

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            should_keep = frame_index % frame_stride == 0 or frame_index == required_frame_index
            if should_keep:
                frame_path = frame_dir / f"{processed_index:06d}.jpg"
                if not cv2.imwrite(str(frame_path), frame):
                    raise RuntimeError(f"Could not write extracted frame: {frame_path}")
                processed_to_original[processed_index] = frame_index
                processed_to_timestamp[processed_index] = round(frame_index / max(source_fps, 1e-6), 6)
                processed_index += 1
                if max_processed_frames is not None and processed_index >= max_processed_frames:
                    frame_index += 1
                    break

            frame_index += 1
    finally:
        cap.release()

    if not processed_to_original:
        raise RuntimeError(f"No frames were extracted from {video_path}")

    return ExtractedVideo(
        frame_dir=frame_dir,
        processed_to_original=processed_to_original,
        processed_to_timestamp=processed_to_timestamp,
        source_fps=source_fps,
        output_fps=max(source_fps / frame_stride, 1.0),
        width=width,
        height=height,
        source_frame_count=frame_index,
    )


def build_video_predictor(
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

    if sam2_checkpoint and sam2_config:
        return build_sam2_video_predictor(
            sam2_config,
            sam2_checkpoint,
            device=str(device),
        )

    model_id = sam2_model_id or os.getenv("SAM2_MODEL_ID", "facebook/sam2.1-hiera-tiny")
    return build_sam2_video_predictor_hf(model_id, device=str(device))


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
