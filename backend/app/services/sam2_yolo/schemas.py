from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np


MODEL_NAME = "yolo+sam2"
PROMPT_SOURCE = "yolo_scissors_bbox_center"
TRACKING_POINT_TYPE = "bbox_center"
TRAJECTORY_POINT_MODE_BBOX_CENTER = "bbox_center"
TRAJECTORY_POINT_MODE_CHOSEN = "chosen_tracking_point"
TRAJECTORY_POINT_MODE_BLADE_TIP = "blade_tip_candidate"
TRAJECTORY_POINT_MODES = (
    TRAJECTORY_POINT_MODE_BBOX_CENTER,
    TRAJECTORY_POINT_MODE_CHOSEN,
    TRAJECTORY_POINT_MODE_BLADE_TIP,
)
DEFAULT_TRAJECTORY_POINT_MODE = TRAJECTORY_POINT_MODE_BLADE_TIP
REGION_RADIUS_PX = 45
OVERLAY_FILENAME = "sam2_yolo_overlay.mp4"
TRAJECTORY_LINE_OVERLAY_FILENAME = "sam2_yolo_trajectory_line_overlay.mp4"
SMOOTH_POLYLINE_OVERLAY_FILENAME = "sam2_yolo_smooth_polyline_overlay.mp4"
RAW_FILENAME = "raw.json"
METRICS_FILENAME = "metrics.json"
SUMMARY_FILENAME = "summary.json"
CLEANED_TRAJECTORY_FILENAME = "cleaned_trajectory.json"
DEBUG_FIRST_FRAME_FILENAME = "debug_first_frame.jpg"
TRACKING_QUALITY_NOTE = (
    "Mask may include hand, but trajectory remains usable because the tracked region follows "
    "the scissors movement."
)


@dataclass(slots=True)
class YoloScissorsDetection:
    bbox: list[float]
    confidence: float
    class_name: str | None = None

    @property
    def point(self) -> list[float]:
        x1, y1, x2, y2 = self.bbox
        return [
            round((x1 + x2) / 2.0, 3),
            round((y1 + y2) / 2.0, 3),
        ]

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["point"] = self.point
        if payload["class_name"] is None:
            payload.pop("class_name")
        return payload


@dataclass(slots=True)
class InitialPrompt:
    frame_index: int
    timestamp_sec: float
    point: list[float]
    bbox: list[float]
    confidence: float
    raw_yolo_bbox: list[float] | None = None
    sam2_prompt_bbox: list[float] | None = None
    bbox_shrink: float | None = None
    class_name: str | None = None

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        if payload["raw_yolo_bbox"] is None:
            payload["raw_yolo_bbox"] = payload["bbox"]
        if payload["sam2_prompt_bbox"] is None:
            payload["sam2_prompt_bbox"] = payload["bbox"]
        if payload["class_name"] is None:
            payload.pop("class_name")
        if payload["bbox_shrink"] is None:
            payload.pop("bbox_shrink")
        return payload


@dataclass(slots=True)
class ExtractedVideo:
    frame_dir: Path
    processed_to_original: dict[int, int]
    processed_to_timestamp: dict[int, float]
    source_fps: float
    output_fps: float
    width: int
    height: int
    source_frame_count: int


def mask_to_metrics(
    mask: np.ndarray,
) -> tuple[int, list[float] | None, list[int] | None, dict[str, list[int]] | None]:
    ys, xs = np.where(mask)
    mask_area = int(xs.size)
    if mask_area == 0:
        return 0, None, None, None

    centroid = [round(float(xs.mean()), 3), round(float(ys.mean()), 3)]
    bbox = [
        int(xs.min()),
        int(ys.min()),
        int(xs.max()),
        int(ys.max()),
    ]
    return mask_area, centroid, bbox, mask_extreme_points(mask)


def mask_extreme_points(mask: np.ndarray) -> dict[str, list[int]] | None:
    mask_uint8 = np.asarray(mask, dtype=np.uint8)
    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    contour = max(contours, key=cv2.contourArea)
    points = contour.reshape(-1, 2)
    if points.size == 0:
        return None

    topmost = points[np.argmin(points[:, 1])]
    bottommost = points[np.argmax(points[:, 1])]
    leftmost = points[np.argmin(points[:, 0])]
    rightmost = points[np.argmax(points[:, 0])]
    return {
        "topmost": [int(topmost[0]), int(topmost[1])],
        "bottommost": [int(bottommost[0]), int(bottommost[1])],
        "leftmost": [int(leftmost[0]), int(leftmost[1])],
        "rightmost": [int(rightmost[0]), int(rightmost[1])],
    }


def bbox_center(bbox: list[int] | None) -> list[float] | None:
    if bbox is None:
        return None
    x1, y1, x2, y2 = bbox
    return [
        round((float(x1) + float(x2)) / 2.0, 3),
        round((float(y1) + float(y2)) / 2.0, 3),
    ]


def build_region(
    *,
    frame_index: int,
    timestamp_sec: float,
    point: list[float] | None,
    valid: bool,
) -> dict[str, Any]:
    if point is None:
        return {
            "frame_index": frame_index,
            "timestamp_sec": round(timestamp_sec, 6),
            "shape": "circle",
            "cx": None,
            "cy": None,
            "r": REGION_RADIUS_PX,
            "source": "sam2_yolo_tracking_point",
            "valid": False,
        }

    return {
        "frame_index": frame_index,
        "timestamp_sec": round(timestamp_sec, 6),
        "shape": "circle",
        "cx": round(float(point[0]), 3),
        "cy": round(float(point[1]), 3),
        "r": REGION_RADIUS_PX,
        "source": "sam2_yolo_tracking_point",
        "valid": valid,
    }


def build_frame_payload(
    *,
    frame_index: int,
    processed_frame_index: int,
    timestamp_sec: float,
    mask_bbox: list[int] | None,
    mask_centroid: list[float] | None,
    mask_area: int,
    mask_extreme_points: dict[str, list[int]] | None = None,
) -> dict[str, Any]:
    center = bbox_center(mask_bbox)
    tracking_valid = mask_area > 0 and center is not None
    tracking_warning = None if tracking_valid else "mask_missing"
    blade_tip_candidate = (
        mask_extreme_points.get("topmost")
        if tracking_valid and mask_extreme_points is not None
        else None
    )
    region = build_region(
        frame_index=frame_index,
        timestamp_sec=timestamp_sec,
        point=center,
        valid=tracking_valid,
    )
    return {
        "frame_index": frame_index,
        "processed_frame_index": processed_frame_index,
        "timestamp_sec": round(timestamp_sec, 6),
        "mask_bbox": mask_bbox,
        "bbox_center": center,
        "mask_centroid": mask_centroid,
        "mask_extreme_points": mask_extreme_points,
        "blade_tip_candidate": blade_tip_candidate,
        "blade_point": None,
        "chosen_tracking_point": center,
        "region": {
            "shape": region["shape"],
            "cx": region["cx"],
            "cy": region["cy"],
            "r": region["r"],
            "source": region["source"],
        },
        "mask_area": mask_area,
        "tracking_valid": tracking_valid,
        "tracking_warning": tracking_warning,
    }


def trajectory_from_frames(frames: list[dict[str, Any]]) -> dict[str, Any]:
    points: list[dict[str, Any]] = []
    for frame in frames:
        point = frame["chosen_tracking_point"]
        points.append(
            {
                "frame_index": frame["frame_index"],
                "timestamp_sec": frame["timestamp_sec"],
                "x": round(float(point[0]), 3) if point is not None else None,
                "y": round(float(point[1]), 3) if point is not None else None,
                "valid": bool(frame["tracking_valid"] and point is not None),
            }
        )
    return {
        "point_type": TRACKING_POINT_TYPE,
        "points": points,
    }


def regions_from_frames(frames: list[dict[str, Any]]) -> list[dict[str, Any]]:
    regions: list[dict[str, Any]] = []
    for frame in frames:
        region = frame["region"]
        regions.append(
            {
                "frame_index": frame["frame_index"],
                "timestamp_sec": frame["timestamp_sec"],
                "shape": region["shape"],
                "cx": region["cx"],
                "cy": region["cy"],
                "r": region["r"],
                "source": region["source"],
                "valid": bool(frame["tracking_valid"] and region["cx"] is not None),
            }
        )
    return regions
