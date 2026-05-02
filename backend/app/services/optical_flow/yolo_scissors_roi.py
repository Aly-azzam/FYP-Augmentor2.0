from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np

from app.services.sam2_yolo.yolo_scissors_detector import (
    DEFAULT_YOLO_SCISSORS_OUTPUT_ROOT,
    YOLO_SCISSORS_RAW_FILENAME,
)


ROIBox = tuple[int, int, int, int]
DetectionFn = Callable[..., Any]


@dataclass(slots=True)
class YoloScissorsROIConfig:
    """Optical Flow-only ROI expansion config for YOLO scissors detections."""

    roi_expand_x: float = 2.2
    roi_expand_y: float = 2.5
    roi_extra_down_ratio: float = 1.5
    roi_extra_up_ratio: float = 0.4
    max_roi_hold_frames: int = 5
    confidence_threshold: float | None = None
    roi_smoothing_enabled: bool = True
    roi_smoothing_alpha: float = 0.65
    artifact_path: str | Path | None = None
    artifact_root: str | Path = DEFAULT_YOLO_SCISSORS_OUTPUT_ROOT


@dataclass(slots=True)
class YoloScissorsROIResult:
    frame_index: int
    video_path: str | None
    original_scissors_bbox: list[float] | None
    expanded_roi_bbox: list[int] | None
    expanded_roi_bbox_raw: list[int] | None
    expanded_roi_bbox_smoothed: list[int] | None
    detection_confidence: float | None
    roi_source: str = "yolo_scissors_expanded"
    roi_found: bool = False
    roi_reused_from_previous: bool = False
    fallback_used: bool = False
    fallback_reason: str | None = None
    detection_class_name: str | None = None

    @property
    def roi_tuple(self) -> ROIBox | None:
        if self.expanded_roi_bbox_smoothed is None:
            return None
        x1, y1, x2, y2 = self.expanded_roi_bbox_smoothed
        return int(x1), int(y1), int(x2), int(y2)

    def to_debug_dict(self) -> dict[str, Any]:
        """Return the JSON/debug shape expected by Optical Flow callers."""
        return asdict(self)


@dataclass(slots=True)
class _HeldROI:
    original_scissors_bbox: list[float]
    expanded_roi_bbox_raw: list[int]
    expanded_roi_bbox_smoothed: list[int]
    detection_confidence: float | None
    detection_class_name: str | None
    last_valid_frame_index: int


def _frame_bounds(frame: np.ndarray) -> tuple[int, int]:
    height, width = frame.shape[:2]
    return int(width), int(height)


def _coerce_bbox(bbox: Sequence[float]) -> tuple[float, float, float, float]:
    if len(bbox) != 4:
        raise ValueError("YOLO scissors bbox must contain exactly four values.")

    x1, y1, x2, y2 = [float(value) for value in bbox]
    if x2 <= x1 or y2 <= y1:
        raise ValueError("YOLO scissors bbox must have positive width and height.")
    return x1, y1, x2, y2


def expand_scissors_bbox_to_roi(
    scissors_bbox: Sequence[float],
    *,
    frame_width: int,
    frame_height: int,
    config: YoloScissorsROIConfig | None = None,
) -> list[int]:
    """
    Expand a YOLO scissors bbox into a larger scissors+hand/wrist ROI.

    The vertical expansion intentionally follows the requested asymmetric
    formula while horizontal expansion is centered around the scissors bbox.
    """
    config = config or YoloScissorsROIConfig()
    x1, y1, x2, y2 = _coerce_bbox(scissors_bbox)

    width = x2 - x1
    height = y2 - y1
    center_x = (x1 + x2) / 2.0

    new_width = width * float(config.roi_expand_x)
    new_x1 = center_x - new_width / 2.0
    new_x2 = center_x + new_width / 2.0
    new_y1 = y1 - height * float(config.roi_extra_up_ratio)
    new_y2 = y2 + height * float(config.roi_extra_down_ratio)

    clamped_x1 = max(0, min(int(round(new_x1)), frame_width - 1))
    clamped_y1 = max(0, min(int(round(new_y1)), frame_height - 1))
    clamped_x2 = max(0, min(int(round(new_x2)), frame_width))
    clamped_y2 = max(0, min(int(round(new_y2)), frame_height))

    if clamped_x2 <= clamped_x1 or clamped_y2 <= clamped_y1:
        raise ValueError("Expanded YOLO scissors ROI is empty after clamping.")

    return [clamped_x1, clamped_y1, clamped_x2, clamped_y2]


@dataclass(slots=True)
class _SharedYoloFrame:
    frame_index: int
    detected: bool
    bbox: list[float] | None
    confidence: float | None
    class_name: str | None = None


def _file_fingerprint(path: Path) -> tuple[int, str] | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    try:
        with path.open("rb") as file:
            while chunk := file.read(1024 * 1024):
                digest.update(chunk)
        return path.stat().st_size, digest.hexdigest()
    except OSError:
        return None


def _load_json(path: Path) -> dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def find_shared_yolo_artifact_for_video(
    *,
    video_path: str | Path,
    artifact_path: str | Path | None = None,
    artifact_root: str | Path = DEFAULT_YOLO_SCISSORS_OUTPUT_ROOT,
) -> tuple[dict[str, Any], Path] | None:
    video = Path(video_path).expanduser().resolve()
    explicit_path = Path(artifact_path).expanduser().resolve() if artifact_path else None
    candidates = [explicit_path] if explicit_path is not None else []
    root = Path(artifact_root).expanduser().resolve()
    if explicit_path is None and root.is_dir():
        candidates.extend(root.glob(f"*/{YOLO_SCISSORS_RAW_FILENAME}"))

    source_fingerprint: tuple[int, str] | None = None
    for candidate in candidates:
        if candidate is None or not candidate.is_file():
            continue
        payload = _load_json(candidate)
        if payload is None:
            continue

        recorded_path = payload.get("video_path")
        if recorded_path:
            recorded = Path(str(recorded_path)).expanduser()
            recorded_resolved = recorded.resolve() if recorded.is_absolute() else recorded
            if recorded_resolved == video:
                return payload, candidate

            if recorded_resolved.is_file():
                source_fingerprint = source_fingerprint or _file_fingerprint(video)
                if source_fingerprint is not None and _file_fingerprint(recorded_resolved) == source_fingerprint:
                    return payload, candidate

    return None


def _shared_frames_from_payload(payload: dict[str, Any]) -> list[_SharedYoloFrame]:
    frames: list[_SharedYoloFrame] = []
    for item in payload.get("frames", []):
        if not isinstance(item, dict):
            continue
        try:
            frames.append(
                _SharedYoloFrame(
                    frame_index=int(item["frame_index"]),
                    detected=bool(item.get("detected", False)),
                    bbox=(
                        [float(value) for value in item["bbox"]]
                        if item.get("bbox") is not None
                        else None
                    ),
                    confidence=(
                        float(item["confidence"])
                        if item.get("confidence") is not None
                        else None
                    ),
                    class_name=item.get("class_name"),
                )
            )
        except (KeyError, TypeError, ValueError):
            continue
    return sorted(frames, key=lambda frame: frame.frame_index)


def _nearest_shared_frame(
    frames: Sequence[_SharedYoloFrame],
    frame_index: int,
) -> _SharedYoloFrame | None:
    if not frames:
        return None
    return min(frames, key=lambda frame: abs(frame.frame_index - int(frame_index)))


def clamp_roi_bbox(
    bbox: Sequence[float],
    *,
    frame_width: int,
    frame_height: int,
) -> list[int]:
    x1, y1, x2, y2 = [float(value) for value in bbox]
    clamped_x1 = max(0, min(int(round(x1)), frame_width - 1))
    clamped_y1 = max(0, min(int(round(y1)), frame_height - 1))
    clamped_x2 = max(0, min(int(round(x2)), frame_width))
    clamped_y2 = max(0, min(int(round(y2)), frame_height))
    if clamped_x2 <= clamped_x1 or clamped_y2 <= clamped_y1:
        raise ValueError("Smoothed YOLO scissors ROI is empty after clamping.")
    return [clamped_x1, clamped_y1, clamped_x2, clamped_y2]


def smooth_roi_bbox(
    current_bbox: Sequence[int],
    previous_smoothed_bbox: Sequence[int] | None,
    *,
    frame_width: int,
    frame_height: int,
    config: YoloScissorsROIConfig | None = None,
) -> list[int]:
    config = config or YoloScissorsROIConfig()
    if previous_smoothed_bbox is None or not config.roi_smoothing_enabled:
        return clamp_roi_bbox(
            current_bbox,
            frame_width=frame_width,
            frame_height=frame_height,
        )

    alpha = float(np.clip(config.roi_smoothing_alpha, 0.0, 1.0))
    current = np.asarray(current_bbox, dtype=np.float32)
    previous = np.asarray(previous_smoothed_bbox, dtype=np.float32)
    smoothed = alpha * current + (1.0 - alpha) * previous
    return clamp_roi_bbox(
        smoothed.tolist(),
        frame_width=frame_width,
        frame_height=frame_height,
    )


def _default_detect_scissors(
    *,
    frame: np.ndarray,
    confidence_threshold: float | None,
) -> Any:
    from app.services.sam2_yolo.yolo_scissors_detector import (  # noqa: PLC0415
        DEFAULT_ROBOFLOW_CONFIDENCE,
        detect_best_scissors_roboflow,
    )

    return detect_best_scissors_roboflow(
        frame=frame,
        confidence_threshold=(
            DEFAULT_ROBOFLOW_CONFIDENCE
            if confidence_threshold is None
            else float(confidence_threshold)
        ),
    )


def get_yolo_scissors_roi_for_frame(
    frame: np.ndarray,
    frame_index: int,
    video_path: str | Path | None = None,
    previous_bbox: Sequence[int] | None = None,
    *,
    previous_original_scissors_bbox: Sequence[float] | None = None,
    previous_detection_confidence: float | None = None,
    previous_detection_class_name: str | None = None,
    previous_expanded_roi_bbox_raw: Sequence[int] | None = None,
    previous_smoothed_bbox: Sequence[int] | None = None,
    frames_since_previous_bbox: int = 0,
    config: YoloScissorsROIConfig | None = None,
    detection_fn: DetectionFn | None = None,
) -> YoloScissorsROIResult:
    """
    Detect scissors with the existing YOLO service and return an expanded ROI.

    This is an Optical Flow adapter only. It does not alter YOLO/SAM2 behavior
    or their model/output contracts.
    """
    config = config or YoloScissorsROIConfig()
    detector = detection_fn or _default_detect_scissors
    video_path_str = str(video_path) if video_path is not None else None

    try:
        detection = detector(
            frame=frame,
            confidence_threshold=config.confidence_threshold,
        )
        original_bbox = [float(value) for value in detection.bbox]
        frame_width, frame_height = _frame_bounds(frame)
        expanded_roi = expand_scissors_bbox_to_roi(
            original_bbox,
            frame_width=frame_width,
            frame_height=frame_height,
            config=config,
        )
        smoothed_roi = smooth_roi_bbox(
            expanded_roi,
            previous_smoothed_bbox,
            frame_width=frame_width,
            frame_height=frame_height,
            config=config,
        )
        return YoloScissorsROIResult(
            frame_index=int(frame_index),
            video_path=video_path_str,
            original_scissors_bbox=original_bbox,
            expanded_roi_bbox=smoothed_roi,
            expanded_roi_bbox_raw=expanded_roi,
            expanded_roi_bbox_smoothed=smoothed_roi,
            detection_confidence=round(float(detection.confidence), 6),
            roi_found=True,
            detection_class_name=getattr(detection, "class_name", None),
        )
    except Exception as exc:  # noqa: BLE001 - fallback must capture detector-specific misses.
        fallback_reason = f"yolo_missing: {exc}"

    hold_limit = max(0, int(config.max_roi_hold_frames))
    if previous_bbox is not None and frames_since_previous_bbox < hold_limit:
        return YoloScissorsROIResult(
            frame_index=int(frame_index),
            video_path=video_path_str,
            original_scissors_bbox=(
                [float(value) for value in previous_original_scissors_bbox]
                if previous_original_scissors_bbox is not None
                else None
            ),
            expanded_roi_bbox=[int(value) for value in previous_bbox],
            expanded_roi_bbox_raw=(
                [int(value) for value in previous_expanded_roi_bbox_raw]
                if previous_expanded_roi_bbox_raw is not None
                else None
            ),
            expanded_roi_bbox_smoothed=[int(value) for value in previous_bbox],
            detection_confidence=previous_detection_confidence,
            roi_source="previous_yolo_bbox",
            roi_found=True,
            roi_reused_from_previous=True,
            fallback_used=True,
            fallback_reason=fallback_reason,
            detection_class_name=previous_detection_class_name,
        )

    return YoloScissorsROIResult(
        frame_index=int(frame_index),
        video_path=video_path_str,
        original_scissors_bbox=None,
        expanded_roi_bbox=None,
        expanded_roi_bbox_raw=None,
        expanded_roi_bbox_smoothed=None,
        detection_confidence=None,
        roi_found=False,
        fallback_used=True,
        fallback_reason=(
            fallback_reason
            if previous_bbox is None
            else f"roi_hold_expired: {fallback_reason}"
        ),
    )


class YoloScissorsROIProvider:
    """Stateful Optical Flow ROI provider with short YOLO-miss hold behavior."""

    def __init__(
        self,
        *,
        video_path: str | Path | None = None,
        config: YoloScissorsROIConfig | None = None,
        detection_fn: DetectionFn | None = None,
    ) -> None:
        self.video_path = video_path
        self.config = config or YoloScissorsROIConfig()
        self.detection_fn = detection_fn
        self._held_roi: _HeldROI | None = None
        self.last_result: YoloScissorsROIResult | None = None

    def detect(self, frame_bgr: np.ndarray, frame_index: int) -> ROIBox | None:
        held = self._held_roi
        result = get_yolo_scissors_roi_for_frame(
            frame=frame_bgr,
            frame_index=frame_index,
            video_path=self.video_path,
            previous_bbox=held.expanded_roi_bbox_smoothed if held is not None else None,
            previous_original_scissors_bbox=(
                held.original_scissors_bbox if held is not None else None
            ),
            previous_detection_confidence=(
                held.detection_confidence if held is not None else None
            ),
            previous_detection_class_name=(
                held.detection_class_name if held is not None else None
            ),
            previous_expanded_roi_bbox_raw=(
                held.expanded_roi_bbox_raw if held is not None else None
            ),
            previous_smoothed_bbox=(
                held.expanded_roi_bbox_smoothed if held is not None else None
            ),
            frames_since_previous_bbox=(
                int(frame_index) - held.last_valid_frame_index
                if held is not None
                else 0
            ),
            config=self.config,
            detection_fn=self.detection_fn,
        )
        self.last_result = result

        if result.roi_found and not result.roi_reused_from_previous:
            if (
                result.original_scissors_bbox is not None
                and result.expanded_roi_bbox_raw is not None
                and result.expanded_roi_bbox_smoothed is not None
            ):
                self._held_roi = _HeldROI(
                    original_scissors_bbox=result.original_scissors_bbox,
                    expanded_roi_bbox_raw=result.expanded_roi_bbox_raw,
                    expanded_roi_bbox_smoothed=result.expanded_roi_bbox_smoothed,
                    detection_confidence=result.detection_confidence,
                    detection_class_name=result.detection_class_name,
                    last_valid_frame_index=int(frame_index),
                )

        return result.roi_tuple

    def last_debug_dict(self) -> dict[str, Any] | None:
        return self.last_result.to_debug_dict() if self.last_result is not None else None

    def close(self) -> None:
        """Keep parity with HandROIDetector; YOLO adapter owns no closeable resource."""
        return None


class SharedYoloScissorsROIProvider:
    """Optical Flow ROI provider backed by shared yolo_scissors_raw.json artifacts."""

    def __init__(
        self,
        *,
        video_path: str | Path,
        config: YoloScissorsROIConfig | None = None,
    ) -> None:
        self.video_path = Path(video_path).expanduser().resolve()
        self.config = config or YoloScissorsROIConfig()
        matched = find_shared_yolo_artifact_for_video(
            video_path=self.video_path,
            artifact_path=self.config.artifact_path,
            artifact_root=self.config.artifact_root,
        )
        self.artifact_path: Path | None = matched[1] if matched is not None else None
        self.frames = _shared_frames_from_payload(matched[0]) if matched is not None else []
        self._held_roi: _HeldROI | None = None
        self.last_result: YoloScissorsROIResult | None = None

        if self.artifact_path is not None:
            print(
                f"[OF][YOLO-ROI] loaded shared artifact={self.artifact_path}",
                flush=True,
            )
        else:
            print(
                f"[OF][YOLO-ROI] shared artifact missing for video={self.video_path}",
                flush=True,
            )

    def detect(self, frame_bgr: np.ndarray, frame_index: int) -> ROIBox | None:
        frame_width, frame_height = _frame_bounds(frame_bgr)
        nearest = _nearest_shared_frame(self.frames, frame_index)
        if nearest is not None and nearest.detected and nearest.bbox is not None:
            original_bbox = [float(value) for value in nearest.bbox]
            expanded_roi = expand_scissors_bbox_to_roi(
                original_bbox,
                frame_width=frame_width,
                frame_height=frame_height,
                config=self.config,
            )
            result = YoloScissorsROIResult(
                frame_index=int(frame_index),
                video_path=str(self.video_path),
                original_scissors_bbox=original_bbox,
                expanded_roi_bbox=expanded_roi,
                expanded_roi_bbox_raw=expanded_roi,
                expanded_roi_bbox_smoothed=expanded_roi,
                detection_confidence=(
                    round(float(nearest.confidence), 6)
                    if nearest.confidence is not None
                    else None
                ),
                roi_source="yolo_scissors_expanded",
                roi_found=True,
                detection_class_name=nearest.class_name,
            )
            self._held_roi = _HeldROI(
                original_scissors_bbox=original_bbox,
                expanded_roi_bbox_raw=expanded_roi,
                expanded_roi_bbox_smoothed=expanded_roi,
                detection_confidence=result.detection_confidence,
                detection_class_name=nearest.class_name,
                last_valid_frame_index=int(frame_index),
            )
            self.last_result = result
            return result.roi_tuple

        held = self._held_roi
        hold_limit = max(0, int(self.config.max_roi_hold_frames))
        if held is not None and int(frame_index) - held.last_valid_frame_index <= hold_limit:
            result = YoloScissorsROIResult(
                frame_index=int(frame_index),
                video_path=str(self.video_path),
                original_scissors_bbox=held.original_scissors_bbox,
                expanded_roi_bbox=held.expanded_roi_bbox_smoothed,
                expanded_roi_bbox_raw=held.expanded_roi_bbox_raw,
                expanded_roi_bbox_smoothed=held.expanded_roi_bbox_smoothed,
                detection_confidence=held.detection_confidence,
                roi_source="previous_yolo_bbox",
                roi_found=True,
                roi_reused_from_previous=True,
                fallback_used=True,
                fallback_reason="shared_yolo_detection_missing",
                detection_class_name=held.detection_class_name,
            )
            self.last_result = result
            return result.roi_tuple

        fallback_reason = (
            "shared_yolo_artifact_missing"
            if self.artifact_path is None
            else "shared_yolo_detection_missing"
        )
        self.last_result = YoloScissorsROIResult(
            frame_index=int(frame_index),
            video_path=str(self.video_path),
            original_scissors_bbox=None,
            expanded_roi_bbox=None,
            expanded_roi_bbox_raw=None,
            expanded_roi_bbox_smoothed=None,
            detection_confidence=None,
            roi_source="none",
            roi_found=False,
            fallback_used=True,
            fallback_reason=fallback_reason,
        )
        return None

    def last_debug_dict(self) -> dict[str, Any] | None:
        return self.last_result.to_debug_dict() if self.last_result is not None else None

    def close(self) -> None:
        return None

