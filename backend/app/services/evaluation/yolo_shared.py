"""Shared YOLO runner for the learner evaluation pipeline.

Runs YOLO once on the learner video (stride=1, every frame) using
backend/models/yolo/best.pt and saves a single yolo_detections.json.

Both the SAM2+YOLO pipeline and the Angle+DTW pipeline read from this
file — neither re-runs YOLO themselves.

Model-loading pattern copied verbatim from:
  app/services/sam2_yolo/yolo_scissors_detector.py (_load_local_yolo_model,
  _select_yolo_device, resolve_local_yolo_weights_path)

Confidence constant copied from:
  app/services/angle/detector.py  (YOLO_CONFIDENCE = 0.25)
"""

from __future__ import annotations

import json
import uuid
from functools import lru_cache
from pathlib import Path
from typing import Any

import cv2

# ── Paths ─────────────────────────────────────────────────────────────────────
# yolo_shared.py → parents[0]=evaluation  parents[1]=services
#                  parents[2]=app         parents[3]=backend
_BACKEND_ROOT = Path(__file__).resolve().parents[3]
_DEFAULT_MODEL_PATH = _BACKEND_ROOT / "models" / "yolo" / "best.pt"

# ── Constants — copied from existing pipelines ────────────────────────────────
# Confidence: same as angle/detector.py
YOLO_CONFIDENCE: float = 0.25

YOLO_DETECTIONS_FILENAME = "yolo_detections.json"


# ── Model loading — copied from sam2_yolo/yolo_scissors_detector.py ───────────

@lru_cache(maxsize=1)
def _load_local_yolo_model(model_path: str) -> Any:
    from ultralytics import YOLO  # noqa: PLC0415

    return YOLO(model_path)


def _select_yolo_device() -> int | str:
    """Return GPU device index 0 if CUDA is available, else 'cpu'.

    Copied from sam2_yolo/yolo_scissors_detector.py.
    """
    try:
        import torch  # type: ignore  # noqa: PLC0415

        if torch.cuda.is_available():
            return 0
    except Exception:  # noqa: BLE001
        pass
    return "cpu"


def _get_model() -> Any:
    if not _DEFAULT_MODEL_PATH.is_file():
        raise RuntimeError(
            f"YOLO weights not found. Expected: {_DEFAULT_MODEL_PATH}"
        )
    return _load_local_yolo_model(str(_DEFAULT_MODEL_PATH))


# ── Per-frame detection ───────────────────────────────────────────────────────

def _detect_frame(frame: Any, model: Any) -> tuple[bool, list[float] | None, float | None]:
    """Run YOLO on a single frame.

    Returns (detected, bbox_xyxy, confidence).
    Copied predict call from sam2_yolo/yolo_scissors_detector.py (detect_best_scissors_local).
    """
    results = model.predict(
        source=frame,
        conf=YOLO_CONFIDENCE,
        device=_select_yolo_device(),
        verbose=False,
    )
    if not results:
        return False, None, None

    result = results[0]
    boxes = getattr(result, "boxes", None)
    if boxes is None or len(boxes) == 0:
        return False, None, None

    names = getattr(result, "names", {}) or {}
    best_bbox: list[float] | None = None
    best_conf: float = -1.0

    scissors_found: list[tuple[list[float], float]] = []
    all_found: list[tuple[list[float], float]] = []

    for box in boxes:
        conf = float(box.conf.detach().cpu().item())
        if conf < YOLO_CONFIDENCE:
            continue
        xyxy = [float(v) for v in box.xyxy.detach().cpu().numpy()[0].tolist()]
        class_id = (
            int(box.cls.detach().cpu().item())
            if getattr(box, "cls", None) is not None
            else -1
        )
        class_name = str(names.get(class_id, "")).strip().lower()
        all_found.append((xyxy, conf))
        if "scissor" in class_name:
            scissors_found.append((xyxy, conf))

    pool = scissors_found or all_found
    if not pool:
        return False, None, None

    best_bbox, best_conf = max(pool, key=lambda item: item[1])
    return True, best_bbox, best_conf


# ── Public API ────────────────────────────────────────────────────────────────

def run_shared_yolo(
    video_path: str,
    output_dir: str,
) -> dict:
    """Run YOLO once on the learner video at stride=1 (every frame).

    Saves yolo_detections.json to ``output_dir`` and returns a dict with
    everything both downstream pipelines need:

    Returns
    -------
    dict with keys:
        run_id                 : str   — unique ID for this YOLO run
        first_valid_frame      : int   — first frame index where scissors detected
        first_valid_bbox       : list  — [x1, y1, x2, y2] of that frame (for SAM2)
        first_valid_bbox_center: list  — [cx, cy] center of that bbox (for SAM2)
        all_detections         : list  — full per-frame detection list
        yolo_detections_path   : str   — absolute path to yolo_detections.json
    """
    video = Path(video_path).expanduser().resolve()
    out_dir = Path(output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    run_id = str(uuid.uuid4())
    model = _get_model()

    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video}")

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    total_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    frames: list[dict] = []
    first_valid_frame: int | None = None
    first_valid_bbox: list[float] | None = None
    first_valid_bbox_center: list[float] | None = None
    frame_index = 0

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if frame is None or frame.size == 0:
                frame_index += 1
                continue

            detected, bbox, conf = _detect_frame(frame, model)

            bbox_center: list[float] | None = None
            if detected and bbox is not None:
                cx = round((bbox[0] + bbox[2]) / 2.0, 3)
                cy = round((bbox[1] + bbox[3]) / 2.0, 3)
                bbox_center = [cx, cy]
                bbox = [round(v, 3) for v in bbox]

                if first_valid_frame is None:
                    first_valid_frame = frame_index
                    first_valid_bbox = bbox
                    first_valid_bbox_center = bbox_center

            frames.append(
                {
                    "frame_index": frame_index,
                    "detected": detected,
                    "bbox": bbox if detected else None,
                    "confidence": round(float(conf), 6) if conf is not None else None,
                    "bbox_center": bbox_center if detected else None,
                }
            )
            frame_index += 1
    finally:
        cap.release()

    total_processed = len(frames)

    # Build JSON payload.
    payload: dict = {
        "run_id": run_id,
        "video_path": str(video),
        "model_path": str(_DEFAULT_MODEL_PATH.relative_to(_BACKEND_ROOT.parent)),
        "stride": 1,
        "fps": round(fps, 6),
        "total_video_frames": total_video_frames,
        "total_processed_frames": total_processed,
        "first_valid_frame": first_valid_frame,
        "first_valid_bbox": first_valid_bbox,
        "first_valid_bbox_center": first_valid_bbox_center,
        "frames": frames,
    }

    out_path = out_dir / YOLO_DETECTIONS_FILENAME
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    # Relative display path for the terminal print.
    try:
        display_path = out_path.relative_to(_BACKEND_ROOT.parent).as_posix()
    except ValueError:
        display_path = str(out_path)

    print(
        f"YOLO run complete — first valid frame: {first_valid_frame} "
        f"— total frames processed: {total_processed}"
    )
    print(f"Saved: {display_path}")

    return {
        "run_id": run_id,
        "first_valid_frame": first_valid_frame,
        "first_valid_bbox": first_valid_bbox,
        "first_valid_bbox_center": first_valid_bbox_center,
        "all_detections": frames,
        "yolo_detections_path": str(out_path),
    }
