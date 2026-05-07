"""YOLO scissors detection and angle extraction.

Logic copied from yolo_scissors_test/app/main.py. Only paths and
HTTPException → RuntimeError adapted for the service layer.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from ultralytics import YOLO

# ── Model path ────────────────────────────────────────────────────────────────
# backend/app/services/angle/detector.py → parents[3] = backend/
_BACKEND_ROOT = Path(__file__).resolve().parents[3]
YOLO_MODEL_PATH = _BACKEND_ROOT / "models" / "yolo" / "best.pt"

# ── Constants (same values as yolo_scissors_test) ─────────────────────────────
YOLO_CONFIDENCE: float = 0.25
ANGLE_JUMP_LIMIT_DEGREES: float = 35.0
LOW_CONFIDENCE_ANGLE_THRESHOLD: float = 0.35
MIN_BLADE_CONTOUR_AREA: float = 40.0
MIN_BLADE_ASPECT_RATIO: float = 2.0

# ── Lazy singleton ─────────────────────────────────────────────────────────────
_yolo_model: YOLO | None = None


def get_yolo_model() -> YOLO:
    global _yolo_model
    if not YOLO_MODEL_PATH.is_file():
        raise RuntimeError(
            f"best.pt does not exist. Put your trained model at {YOLO_MODEL_PATH}"
        )
    if _yolo_model is None:
        _yolo_model = YOLO(str(YOLO_MODEL_PATH))
    return _yolo_model


# ── Detection ─────────────────────────────────────────────────────────────────

def detect_scissors(frame: Any, model: YOLO) -> list[dict]:
    try:
        results = model.predict(source=frame, conf=YOLO_CONFIDENCE, verbose=False)
    except Exception as exc:
        raise RuntimeError(f"YOLO inference error: {str(exc)[:300]}") from exc

    if not results:
        return []

    result = results[0]
    boxes = getattr(result, "boxes", None)
    if boxes is None or len(boxes) == 0:
        return []

    names = _result_names(result, model)
    xyxy_values = boxes.xyxy.detach().cpu().numpy()
    confidence_values = boxes.conf.detach().cpu().numpy()
    class_values = (
        boxes.cls.detach().cpu().numpy()
        if boxes.cls is not None
        else [None] * len(boxes)
    )

    predictions: list[dict] = []
    for bbox, conf, class_id in zip(xyxy_values, confidence_values, class_values):
        x1, y1, x2, y2 = [float(value) for value in bbox.tolist()]
        predictions.append(
            {
                "class": _class_name(names, class_id) or "scissors",
                "confidence": float(conf),
                "bbox": [x1, y1, x2, y2],
                "width": max(0.0, x2 - x1),
                "height": max(0.0, y2 - y1),
            }
        )

    return predictions


def pick_best_scissors(
    predictions: list[dict],
    frame_width: int,
    frame_height: int,
) -> tuple[dict | None, list[dict]]:
    frame_area = max(1.0, float(frame_width * frame_height))
    raw_debug: list[dict] = []
    class_names = [str(pred.get("class", "")).lower().strip() for pred in predictions]
    has_scissors_class = any("scissor" in class_name for class_name in class_names)
    candidates: list[dict] = []

    for pred in predictions:
        cls = str(pred.get("class", "")).lower().strip()
        conf = float(pred.get("confidence", 0.0))
        bbox = [float(value) for value in pred["bbox"]]
        w = max(0.0, float(pred.get("width", 0.0)))
        h = max(0.0, float(pred.get("height", 0.0)))
        area_ratio = (w * h) / frame_area
        aspect_ratio = h / max(w, 1e-6)
        accepted = conf >= YOLO_CONFIDENCE and (
            "scissor" in cls or not has_scissors_class
        )
        rejection_reason = None
        if conf < YOLO_CONFIDENCE:
            rejection_reason = "low_confidence"
        elif has_scissors_class and "scissor" not in cls:
            rejection_reason = "not_scissors"

        raw_debug.append(
            {
                "class": pred.get("class", ""),
                "confidence": round(conf, 6),
                "bbox": [round(v, 3) for v in bbox],
                "area_ratio": round(area_ratio, 6),
                "aspect_ratio": round(aspect_ratio, 6),
                "accepted": accepted,
                "rejection_reason": rejection_reason,
            }
        )

        if accepted:
            candidates.append(pred)

    best = max(
        candidates,
        key=lambda pred: float(pred.get("confidence", 0.0)),
        default=None,
    )
    return best, raw_debug


# ── Angle extraction ───────────────────────────────────────────────────────────

def estimate_blade_angle_from_crop(
    frame: Any,
    bbox: tuple[int, int, int, int],
) -> float | None:
    frame_height, frame_width = frame.shape[:2]
    x1, y1, x2, y2 = bbox
    x1 = max(0, min(frame_width - 1, x1))
    x2 = max(0, min(frame_width, x2))
    y1 = max(0, min(frame_height - 1, y1))
    y2 = max(0, min(frame_height, y2))

    if x2 <= x1 or y2 <= y1:
        return None

    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return None

    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)

    metal_lower = np.array([0, 0, 100])
    metal_upper = np.array([180, 80, 255])
    metal_mask = cv2.inRange(hsv, metal_lower, metal_upper)

    skin_lower = np.array([0, 30, 60])
    skin_upper = np.array([25, 180, 255])
    skin_mask_1 = cv2.inRange(hsv, skin_lower, skin_upper)
    skin_mask_2 = cv2.inRange(hsv, np.array([160, 30, 60]), np.array([180, 180, 255]))
    skin_mask = cv2.bitwise_or(skin_mask_1, skin_mask_2)
    mask = cv2.bitwise_and(metal_mask, cv2.bitwise_not(skin_mask))

    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour = choose_longest_blade_contour(contours)
    if contour is None:
        return None

    points = contour.reshape(-1, 2).astype(np.float32)
    points[:, 0] += x1
    points[:, 1] += y1
    if len(points) < 2:
        return None

    vx, vy, _, _ = cv2.fitLine(points, cv2.DIST_L2, 0, 0.01, 0.01).flatten()
    return normalize_angle_180(math.degrees(math.atan2(float(vy), float(vx))))


def choose_longest_blade_contour(contours: tuple[Any, ...]) -> Any | None:
    best_contour = None
    best_length = 0.0

    for contour in contours:
        area = cv2.contourArea(contour)
        if area <= MIN_BLADE_CONTOUR_AREA:
            continue

        (_, _), (width, height), _ = cv2.minAreaRect(contour)
        short_side = min(width, height)
        long_side = max(width, height)
        if short_side <= 0:
            continue

        aspect_ratio = long_side / short_side
        if aspect_ratio <= MIN_BLADE_ASPECT_RATIO:
            continue

        if long_side > best_length:
            best_contour = contour
            best_length = long_side

    return best_contour


def extend_angle_line(
    angle_degrees: float,
    center_x: float,
    center_y: float,
    frame_width: int,
    frame_height: int,
    scale: int = 2,
) -> tuple[tuple[int, int], tuple[int, int]]:
    length = max(frame_width, frame_height) * scale
    angle_radians = math.radians(angle_degrees)
    ux = math.cos(angle_radians)
    uy = math.sin(angle_radians)

    start = (int(center_x - ux * length), int(center_y - uy * length))
    end = (int(center_x + ux * length), int(center_y + uy * length))

    return start, end


def angle_from_points(p1: tuple[int, int], p2: tuple[int, int]) -> float:
    x1, y1 = p1
    x2, y2 = p2
    return normalize_angle_180(math.degrees(math.atan2(y2 - y1, x2 - x1)))


def normalize_angle_180(angle: float) -> float:
    return angle % 180.0


def angle_delta_degrees(previous_angle: float, current_angle: float) -> float:
    return ((current_angle - previous_angle + 90.0) % 180.0) - 90.0


def smooth_angle(previous_angle: float, current_angle: float) -> float:
    delta = angle_delta_degrees(previous_angle, current_angle)
    return normalize_angle_180(previous_angle + 0.6 * delta)


# ── JSON payload helpers ───────────────────────────────────────────────────────

def build_video_frames_payload(
    *,
    video_type: str,
    video_path: str,
    total_frames: int,
    fps: float,
    frames: list[dict],
) -> dict:
    return {
        "video_type": video_type,
        "video_path": video_path,
        "total_frames": total_frames,
        "fps": round(float(fps), 6),
        "frames": [format_video_frame_record(frame, fps) for frame in frames],
    }


def format_video_frame_record(frame: dict, fps: float) -> dict:
    frame_index = int(frame["frame_index"])
    valid_line = bool(frame.get("valid_line"))
    bbox = bbox_to_dict(frame["bbox"]) if frame.get("bbox") is not None else None
    line_center = (
        point_to_dict(frame["line_center"])
        if frame.get("line_center") is not None
        else None
    )
    line_start = (
        point_to_dict(frame["line_start"])
        if frame.get("line_start") is not None
        else None
    )
    line_end = (
        point_to_dict(frame["line_end"])
        if frame.get("line_end") is not None
        else None
    )
    detected = bbox is not None

    if not detected or not valid_line:
        return {
            "frame_index": frame_index,
            "timestamp_sec": frame_timestamp(frame_index, fps),
            "detected": detected,
            "valid_line": False,
            "confidence": round_optional(frame.get("confidence")) if detected else None,
            "bbox": bbox if detected else None,
            "line_center": None,
            "line_angle": None,
            "line_start": None,
            "line_end": None,
            "line_source": "none",
            "fallback_used": False,
        }

    return {
        "frame_index": frame_index,
        "timestamp_sec": frame_timestamp(frame_index, fps),
        "detected": detected,
        "valid_line": True,
        "confidence": round_optional(frame.get("confidence")),
        "bbox": bbox,
        "line_center": line_center,
        "line_angle": round_optional(frame.get("line_angle")),
        "line_start": line_start,
        "line_end": line_end,
        "line_source": frame.get("line_source") or "none",
        "fallback_used": bool(frame.get("fallback_used", False)),
    }


def bbox_to_dict(bbox: list[float] | tuple[float, ...]) -> dict:
    x1, y1, x2, y2 = [float(value) for value in bbox]
    return {
        "x1": round(x1, 3),
        "y1": round(y1, 3),
        "x2": round(x2, 3),
        "y2": round(y2, 3),
        "width": round(max(0.0, x2 - x1), 3),
        "height": round(max(0.0, y2 - y1), 3),
    }


def point_to_dict(point: list[float] | tuple[float, ...]) -> dict:
    x, y = [float(value) for value in point]
    return {
        "x": round(x, 3),
        "y": round(y, 3),
    }


def round_optional(value: Any, digits: int = 6) -> float | None:
    if value is None:
        return None
    return round(float(value), digits)


def frame_timestamp(frame_index: int, fps: float) -> float:
    return round(float(frame_index) / max(float(fps), 1e-6), 6)


def _result_names(result: Any, model: YOLO) -> dict[int, str]:
    names = getattr(result, "names", None) or getattr(model, "names", None) or {}
    return {int(key): str(value) for key, value in dict(names).items()}


def _class_name(names: dict[int, str], class_id: Any) -> str | None:
    if class_id is None:
        return None
    return names.get(int(class_id), str(int(class_id)))
