from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np


ROIBox = tuple[int, int, int, int]
DEFAULT_LOCAL_YOLO_SCISSORS_MODEL_PATH = (
    Path(__file__).resolve().parents[3]
    / "models"
    / "yolo"
    / "best.pt"
)


@dataclass(slots=True)
class YoloScissorsROIConfig:
    """Optical Flow-only ROI expansion config for YOLO scissors detections."""

    roi_expand_x: float = 2.4
    roi_expand_y: float = 2.8
    roi_extra_x_ratio: float = 0.7
    roi_extra_down_ratio: float = 1.5
    roi_extra_up_ratio: float = 0.3
    max_roi_hold_frames: int = 5
    confidence_threshold: float | None = 0.25
    roi_smoothing_enabled: bool = False
    roi_smoothing_alpha: float = 0.65
    model_path: str | Path = DEFAULT_LOCAL_YOLO_SCISSORS_MODEL_PATH
    imgsz: int = 640
    half: bool = True


@dataclass(slots=True)
class YoloScissorsROIResult:
    frame_index: int
    video_path: str | None
    original_scissors_bbox: list[float] | None
    expanded_roi_bbox: list[int] | None
    expanded_roi_bbox_raw: list[int] | None
    expanded_roi_bbox_smoothed: list[int] | None
    detection_confidence: float | None
    roi_source: str = "yolo"
    roi_found: bool = False
    roi_reused_from_previous: bool = False
    fallback_used: bool = False
    fallback_reason: str | None = None
    detection_class_name: str | None = None
    yolo_detected: bool = False

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
class YoloScissorsFrameDetection:
    frame_index: int
    detected: bool
    bbox: list[float] | None
    expanded_roi: list[int] | None
    confidence: float | None
    class_name: str | None = None


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


def expand_scissors_bbox_to_hand_roi(
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    frame_width: int,
    frame_height: int,
) -> list[int]:
    bw = float(x2) - float(x1)
    bh = float(y2) - float(y1)
    if bw <= 0 or bh <= 0:
        raise ValueError("YOLO scissors bbox must have positive width and height.")

    clamped_x1 = max(0, int(float(x1) - 0.7 * bw))
    clamped_x2 = min(int(frame_width), int(float(x2) + 0.7 * bw))
    clamped_y1 = max(0, int(float(y1) - 0.3 * bh))
    clamped_y2 = min(int(frame_height), int(float(y2) + 1.5 * bh))

    if clamped_x2 <= clamped_x1 or clamped_y2 <= clamped_y1:
        raise ValueError("Expanded YOLO scissors ROI is empty after clamping.")

    return [clamped_x1, clamped_y1, clamped_x2, clamped_y2]


def expand_scissors_bbox_to_roi(
    scissors_bbox: Sequence[float],
    *,
    frame_width: int,
    frame_height: int,
    config: YoloScissorsROIConfig | None = None,
) -> list[int]:
    del config
    x1, y1, x2, y2 = _coerce_bbox(scissors_bbox)
    return expand_scissors_bbox_to_hand_roi(
        x1,
        y1,
        x2,
        y2,
        frame_width,
        frame_height,
    )


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


def _select_yolo_device() -> int | str:
    try:
        import torch  # type: ignore  # noqa: PLC0415

        if torch.cuda.is_available():
            return 0
    except Exception:  # noqa: BLE001 - torch availability varies by environment.
        pass
    return "cpu"


def _best_box_from_result(result: Any) -> tuple[list[float], float, str | None] | None:
    boxes = getattr(result, "boxes", None)
    if boxes is None or len(boxes) == 0:
        return None

    best: tuple[list[float], float, str | None] | None = None
    names = getattr(result, "names", {})
    for box in boxes:
        conf_value = float(box.conf.detach().cpu().item())
        xyxy = [float(value) for value in box.xyxy.detach().cpu().numpy()[0].tolist()]
        class_id = (
            int(box.cls.detach().cpu().item())
            if getattr(box, "cls", None) is not None
            else -1
        )
        class_name = names.get(class_id) if isinstance(names, dict) else None
        if best is None or conf_value > best[1]:
            best = (xyxy, conf_value, class_name)
    return best


def collect_yolo_scissors_detections_for_video(
    *,
    video_path: str | Path,
    config: YoloScissorsROIConfig | None = None,
) -> dict[int, YoloScissorsFrameDetection]:
    """
    Run local YOLO once over the whole video using Ultralytics streaming.

    Frame indices are zero-based to match Ultralytics video result ordering and
    OpenCV frame positions. The Optical Flow pair for frame N uses detection N.
    """
    config = config or YoloScissorsROIConfig()
    model_path = Path(config.model_path).expanduser().resolve()
    print(f"[YOLO-OF] Using model path: {model_path}", flush=True)
    if not model_path.is_file():
        raise FileNotFoundError(f"Local YOLO scissors model not found: {model_path}")

    from ultralytics import YOLO  # noqa: PLC0415

    device = _select_yolo_device()
    use_half = bool(config.half and device != "cpu")
    model = YOLO(str(model_path))
    print(
        f"[YOLO-OF] Loaded YOLO model from {model_path} on device={device}",
        flush=True,
    )
    print(f"[YOLO-OF] Processing video: {video_path}", flush=True)

    started = time.perf_counter()
    detections: dict[int, YoloScissorsFrameDetection] = {}
    detection_count = 0
    results_generator = model.predict(
        source=str(video_path),
        stream=True,
        imgsz=int(config.imgsz),
        conf=float(config.confidence_threshold or 0.25),
        device=device,
        half=use_half,
        verbose=False,
    )

    for frame_index, result in enumerate(results_generator):
        orig_shape = getattr(result, "orig_shape", None)
        if orig_shape is None:
            frame_height, frame_width = 1, 1
        else:
            frame_height, frame_width = int(orig_shape[0]), int(orig_shape[1])

        best = _best_box_from_result(result)
        if best is None:
            detections[frame_index] = YoloScissorsFrameDetection(
                frame_index=frame_index,
                detected=False,
                bbox=None,
                expanded_roi=None,
                confidence=None,
            )
            continue

        bbox, confidence, class_name = best
        expanded_roi = expand_scissors_bbox_to_roi(
            bbox,
            frame_width=frame_width,
            frame_height=frame_height,
            config=config,
        )
        detections[frame_index] = YoloScissorsFrameDetection(
            frame_index=frame_index,
            detected=True,
            bbox=bbox,
            expanded_roi=expanded_roi,
            confidence=round(float(confidence), 6),
            class_name=class_name,
        )
        detection_count += 1

    elapsed = time.perf_counter() - started
    total = len(detections)
    ratio = detection_count / total if total else 0.0
    print(
        "[YOLO-OF] Streaming YOLO done. "
        f"frames={total}, detections={detection_count}, "
        f"yolo_detection_ratio={ratio:.6f}, yolo_runtime_sec={elapsed:.2f}",
        flush=True,
    )
    return detections


class YoloScissorsROIProvider:
    """Deprecated compatibility wrapper.

    Optical Flow must use ``collect_yolo_scissors_detections_for_video`` so YOLO
    receives the whole video with ``stream=True`` instead of individual frames.
    """

    def __init__(
        self,
        *,
        video_path: str | Path | None = None,
        config: YoloScissorsROIConfig | None = None,
    ) -> None:
        self.video_path = str(video_path) if video_path is not None else None
        self.config = config or YoloScissorsROIConfig()
        self.last_result: YoloScissorsROIResult | None = None

    def _detect_best_box(self, frame_bgr: np.ndarray) -> tuple[list[float], float, str | None] | None:
        del frame_bgr
        raise RuntimeError("Use streamed whole-video YOLO detection for Optical Flow.")

    def detect(self, frame_bgr: np.ndarray, frame_index: int) -> ROIBox | None:
        del frame_bgr, frame_index
        raise RuntimeError("Use streamed whole-video YOLO detection for Optical Flow.")

    def last_debug_dict(self) -> dict[str, Any] | None:
        return self.last_result.to_debug_dict() if self.last_result is not None else None

    def close(self) -> None:
        """Keep parity with HandROIDetector; YOLO adapter owns no closeable resource."""
        return None


SharedYoloScissorsROIProvider = YoloScissorsROIProvider


def get_yolo_scissors_roi_for_frame(*_args: Any, **_kwargs: Any) -> YoloScissorsROIResult:
    raise RuntimeError("Use YoloScissorsROIProvider so the local YOLO model is loaded once.")

