"""Offline one-time script to extract and save SAM2 scissor masks for an expert.

Usage (from backend/):
    python -m app.scripts.sam2_yolo.export_expert_masks --expert_id <expert_id>

Reads:
  storage/outputs/sam2_yolo/experts/<expert_id>/metadata.json  -> source_video_path
  storage/outputs/sam2_yolo/experts/<expert_id>/raw.json       -> frames[0] prompt

Saves:
  storage/outputs/sam2_yolo/experts/<expert_id>/masks/frame_<N>.png  (grayscale 0=bg 255=mask)
  storage/outputs/sam2_yolo/experts/<expert_id>/masks/masks_index.json

No app imports — fully standalone.
"""

from __future__ import annotations

import argparse
import contextlib
import gc
import json
import os
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

BACKEND_ROOT = Path(__file__).resolve().parents[3]   # .../backend/
STORAGE_ROOT = BACKEND_ROOT / "storage"
EXPERTS_ROOT = STORAGE_ROOT / "outputs" / "sam2_yolo" / "experts"

_SAM2_CHECKPOINT = BACKEND_ROOT / "models" / "sam2" / "sam2_hiera_tiny.pt"
_SAM2_CONFIG = "configs/sam2/sam2_hiera_t.yaml"
_FRAME_STRIDE = 5

# ---------------------------------------------------------------------------
# Helpers inlined from stable_runner.py (no app imports)
# ---------------------------------------------------------------------------

_CPU_STABILITY_CONFIGURED = False


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


def _choose_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    _configure_cpu_stability()
    return torch.device("cpu")


def _ensure_sam2_on_path() -> None:
    vendor = BACKEND_ROOT / "vendor" / "sam2"
    if vendor.is_dir():
        vendor_str = str(vendor)
        if vendor_str not in sys.path:
            sys.path.insert(0, vendor_str)


def _build_predictor(device: torch.device):
    _ensure_sam2_on_path()
    from sam2.build_sam import build_sam2_video_predictor  # noqa: PLC0415

    return build_sam2_video_predictor(
        _SAM2_CONFIG,
        str(_SAM2_CHECKPOINT),
        device=str(device),
    )


def _autocast(device: torch.device) -> contextlib.AbstractContextManager:
    if device.type == "cuda":
        return torch.autocast("cuda", dtype=torch.bfloat16)
    return contextlib.nullcontext()


def _cleanup(predictor, inference_state) -> None:
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


def _extract_frames(
    video_path: Path,
    frame_dir: Path,
    stride: int,
    required_frame_index: int,
) -> dict[int, int]:
    """Extract every stride-th frame (always including required_frame_index).

    Returns processed_to_original: {processed_idx: original_idx}.
    """
    if frame_dir.exists():
        shutil.rmtree(frame_dir)
    frame_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    processed_to_original: dict[int, int] = {}
    original_idx = 0
    processed_idx = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            is_required = original_idx == required_frame_index
            if original_idx % stride == 0 or is_required:
                out_path = frame_dir / f"{processed_idx:06d}.jpg"
                cv2.imwrite(str(out_path), frame)
                processed_to_original[processed_idx] = original_idx
                processed_idx += 1
            original_idx += 1
    finally:
        cap.release()

    if not processed_to_original:
        raise RuntimeError(f"No frames extracted from {video_path}")
    if required_frame_index not in processed_to_original.values():
        raise RuntimeError(f"Prompt frame {required_frame_index} not in extracted frames")

    return dict(sorted(processed_to_original.items()))


def _processed_for_original(processed_to_original: dict[int, int], original: int) -> int:
    for pi, oi in processed_to_original.items():
        if oi == original:
            return pi
    raise RuntimeError(f"Frame {original} not found in extracted frames")


def _mask_for_object(object_ids, mask_logits: torch.Tensor, obj_id: int = 1) -> np.ndarray:
    id_list = [int(x) for x in object_ids]
    if obj_id not in id_list:
        h, w = mask_logits.shape[-2:]
        return np.zeros((h, w), dtype=bool)
    idx = id_list.index(obj_id)
    mask = (mask_logits[idx] > 0.0).detach().cpu().numpy()
    if mask.ndim == 3:
        mask = mask[0]
    return mask.astype(bool)


# ---------------------------------------------------------------------------
# Main logic
# ---------------------------------------------------------------------------

def export_expert_masks(expert_id: str) -> None:
    expert_dir = EXPERTS_ROOT / expert_id
    metadata_path = expert_dir / "metadata.json"
    raw_path = expert_dir / "raw.json"

    for p in (metadata_path, raw_path):
        if not p.is_file():
            print(f"[ERROR] not found: {p}")
            sys.exit(1)

    source_video_path = Path(json.loads(metadata_path.read_text())["source_video_path"])
    if not source_video_path.is_file():
        print(f"[ERROR] source video not found: {source_video_path}")
        sys.exit(1)

    raw_data = json.loads(raw_path.read_text())
    f0 = raw_data["frames"][0]
    frame_index = int(f0["frame_index"])
    point = list(f0["bbox_center"])        # [x, y]
    box = list(f0["mask_bbox"])            # [x1, y1, x2, y2]

    masks_dir = expert_dir / "masks"
    masks_dir.mkdir(parents=True, exist_ok=True)
    frames_tmp = expert_dir / "_frames_tmp"

    print(f"[export_expert_masks] expert_id={expert_id}")
    print(f"[export_expert_masks] video={source_video_path}")
    print(f"[export_expert_masks] prompt: frame={frame_index}, point={point}, box={box}")

    device = _choose_device()
    print(f"[export_expert_masks] device={device}")

    print(f"[export_expert_masks] extracting frames (stride={_FRAME_STRIDE})...")
    processed_to_original = _extract_frames(
        video_path=source_video_path,
        frame_dir=frames_tmp,
        stride=_FRAME_STRIDE,
        required_frame_index=frame_index,
    )
    print(f"[export_expert_masks] {len(processed_to_original)} processed frames")

    predictor = None
    inference_state = None
    saved_indices: list[int] = []

    try:
        print("[export_expert_masks] loading SAM2...")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.cuda.empty_cache()
        predictor = _build_predictor(device)

        np_point = np.array([point], dtype=np.float32)
        np_label = np.array([1], dtype=np.int32)
        np_box = np.array(box, dtype=np.float32)
        prompt_processed = _processed_for_original(processed_to_original, frame_index)

        with torch.inference_mode(), _autocast(device):
            inference_state = predictor.init_state(
                video_path=str(frames_tmp),
                async_loading_frames=True,
            )
            predictor.add_new_points_or_box(
                inference_state=inference_state,
                frame_idx=prompt_processed,
                obj_id=1,
                points=np_point,
                labels=np_label,
                box=np_box,
                normalize_coords=True,
            )

            print("[export_expert_masks] propagating...")
            for count, (proc_idx, object_ids, mask_logits) in enumerate(
                predictor.propagate_in_video(inference_state), start=1
            ):
                original_idx = processed_to_original[int(proc_idx)]
                mask = _mask_for_object(object_ids, mask_logits, obj_id=1)
                mask_img = (mask * 255).astype(np.uint8)
                out_path = masks_dir / f"frame_{original_idx:06d}.png"
                cv2.imwrite(str(out_path), mask_img)
                saved_indices.append(original_idx)

                if count % 10 == 0:
                    print(f"[export_expert_masks] {count} frames processed...")

    finally:
        _cleanup(predictor, inference_state)
        shutil.rmtree(frames_tmp, ignore_errors=True)

    index_path = masks_dir / "masks_index.json"
    index_path.write_text(
        json.dumps({"expert_id": expert_id, "frame_indices": sorted(saved_indices)}, indent=2)
    )

    print(f"[export_expert_masks] done — {len(saved_indices)} masks saved to {masks_dir}")
    print(f"[export_expert_masks] index -> {index_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export SAM2 scissor masks for an expert video (run once offline)."
    )
    parser.add_argument("--expert_id", required=True, help="Expert UUID (subdirectory under experts/)")
    args = parser.parse_args()
    export_expert_masks(args.expert_id)


if __name__ == "__main__":
    main()
