from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Literal

import cv2
import numpy as np

from .feature_extractor import extract_frame_flow_features
from .schemas import FrameFlowFeatures, VideoMetadata


ROISource = Literal["none", "mediapipe_hand", "yolo_scissors", "yolo_scissors_expanded"]


@dataclass
class FarnebackConfig:
    pyr_scale: float = 0.5
    levels: int = 3
    winsize: int = 15
    iterations: int = 3
    poly_n: int = 5
    poly_sigma: float = 1.2
    flags: int = 0
    motion_threshold: float = 2.0
    resize_width: int | None = None
    resize_height: int | None = None
    #: Odd kernel size for cv2.GaussianBlur before flow, e.g. 5 -> (5, 5). 0 = disabled.
    gaussian_blur_kernel: int = 5
    #: When True, run Farneback only inside a MediaPipe hand bounding box.
    use_hand_roi: bool = False
    #: ROI provider. ``use_hand_roi=True`` keeps legacy MediaPipe behavior unless this is changed.
    roi_source: ROISource = "mediapipe_hand"
    #: Padding added around the detected hand bounding box, in pixels.
    roi_padding_px: int = 40
    #: Minimum effective ROI padding used for locked hand tracking.
    roi_enlarge_padding_px: int = 50
    #: Which hand to track for ROI cropping: right, left, largest, or first.
    roi_hand_preference: str = "right"
    #: Lock onto this target hand; use "right" to avoid switching to the left hand.
    roi_lock_target: str | bool = "right"
    #: Number of frames to reuse locked ROI before strict/loose fallback logic.
    roi_lock_max_missing_frames: int = 5
    #: Maximum accepted center movement as a ratio of frame width.
    roi_lock_max_center_distance_ratio: float = 0.3
    #: If True, never switch to another hand after lock is lost.
    roi_lock_strict: bool = True
    #: Horizontal expansion factor for YOLO scissors Optical Flow ROI.
    roi_expand_x: float = 2.2
    #: Kept for config/debug parity with the YOLO ROI helper.
    roi_expand_y: float = 2.5
    #: Extra downward YOLO ROI expansion as a scissors-box height ratio.
    roi_extra_down_ratio: float = 1.5
    #: Extra upward YOLO ROI expansion as a scissors-box height ratio.
    roi_extra_up_ratio: float = 0.4
    #: Number of frames to reuse the last valid YOLO ROI after a YOLO miss.
    max_roi_hold_frames: int = 5
    #: Optional per-run YOLO confidence threshold; None uses the existing YOLO service default.
    yolo_confidence_threshold: float | None = None
    #: Optional path to a shared yolo_scissors_raw.json artifact.
    yolo_scissors_artifact_path: str | None = None
    #: Smooth YOLO expanded ROI before using it for Optical Flow cropping.
    roi_smoothing_enabled: bool = True
    #: Exponential smoothing alpha for YOLO expanded ROI.
    roi_smoothing_alpha: float = 0.65


def _effective_roi_source(config: FarnebackConfig) -> ROISource:
    roi_source = str(config.roi_source or "none").strip().lower()
    if roi_source not in {"none", "mediapipe_hand", "yolo_scissors", "yolo_scissors_expanded"}:
        raise ValueError(
            "roi_source must be one of: none, mediapipe_hand, yolo_scissors, yolo_scissors_expanded."
        )
    if roi_source == "none":
        return "none"
    if roi_source in {"yolo_scissors", "yolo_scissors_expanded"}:
        return "yolo_scissors_expanded"
    return "mediapipe_hand" if config.use_hand_roi else "none"


def _roi_area_ratio(roi: tuple[int, int, int, int] | None, frame_shape: tuple[int, ...]) -> float:
    if roi is None:
        return 0.0
    x1, y1, x2, y2 = roi
    height, width = frame_shape[:2]
    frame_area = max(1, int(width) * int(height))
    roi_area = max(0, x2 - x1) * max(0, y2 - y1)
    return float(roi_area / frame_area)


def _validate_video_path(video_path: str | Path) -> Path:
    path = Path(video_path)
    if not path.exists():
        raise FileNotFoundError(f"Video not found: {path}")
    if not path.is_file():
        raise FileNotFoundError(f"Path is not a file: {path}")
    return path


def _resize_frame_if_needed(
    frame: np.ndarray,
    resize_width: int | None,
    resize_height: int | None,
) -> np.ndarray:
    if resize_width is None or resize_height is None:
        return frame

    return cv2.resize(
        frame,
        (resize_width, resize_height),
        interpolation=cv2.INTER_AREA,
    )


def _to_gray(frame: np.ndarray) -> np.ndarray:
    if frame.ndim == 2:
        return frame
    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


def _blur_gray_for_flow(gray: np.ndarray, kernel_size: int) -> np.ndarray:
    """
    Light Gaussian blur on grayscale frames before Farneback to reduce sensor noise.
    """
    if kernel_size <= 0:
        return gray
    k = int(kernel_size)
    if k % 2 == 0:
        k += 1
    if k < 3:
        return gray
    return cv2.GaussianBlur(gray, (k, k), 0)


def read_video_metadata(video_path: str | Path) -> VideoMetadata:
    """
    Read basic metadata from a video file.
    """
    path = _validate_video_path(video_path)

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {path}")

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)

    cap.release()

    duration_sec = float(frame_count / fps) if fps > 0 else 0.0

    return VideoMetadata(
        video_path=str(path),
        fps=fps if fps > 0 else 1.0,
        frame_count=max(frame_count, 0),
        duration_sec=round(duration_sec, 6),
        width=max(width, 1),
        height=max(height, 1),
    )


def compute_video_optical_flow_features(
    video_path: str | Path,
    config: FarnebackConfig | None = None,
) -> tuple[VideoMetadata, List[FrameFlowFeatures]]:
    """
    Compute frame-by-frame Farneback optical flow summary features for one video.

    Returns:
        (video_metadata, frame_features)
    """
    config = config or FarnebackConfig()
    path = _validate_video_path(video_path)

    metadata = read_video_metadata(path)

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {path}")
    print("[OF] video opened", flush=True)
    print(f"[OF] fps={metadata.fps}", flush=True)
    print(f"[OF] total_frames={metadata.frame_count}", flush=True)
    print(f"[OF] duration_sec={metadata.duration_sec}", flush=True)
    print(f"[OF] width={metadata.width}", flush=True)
    print(f"[OF] height={metadata.height}", flush=True)

    frame_features: List[FrameFlowFeatures] = []

    success, prev_frame = cap.read()
    if not success or prev_frame is None:
        cap.release()
        return metadata, frame_features

    prev_frame = _resize_frame_if_needed(
        prev_frame,
        config.resize_width,
        config.resize_height,
    )
    prev_gray = _to_gray(prev_frame)

    fps = metadata.fps if metadata.fps > 0 else 1.0
    frame_index = 1
    roi_detector = None
    active_roi_source = _effective_roi_source(config)
    crop_to_roi = None
    embed_roi_flow_in_canvas = None
    processed_count = 0
    yolo_roi_frames = 0
    extraction_start = time.perf_counter()

    if active_roi_source != "none":
        from .hand_roi import (
            HandROIDetector,
            crop_to_roi,
            embed_roi_flow_in_canvas,
        )

        if active_roi_source == "mediapipe_hand":
            effective_roi_padding_px = max(
                int(config.roi_padding_px),
                int(config.roi_enlarge_padding_px),
            )
            roi_detector = HandROIDetector(
                padding_px=effective_roi_padding_px,
                hand_preference=config.roi_hand_preference,
                lock_target=config.roi_lock_target,
                lock_max_missing_frames=config.roi_lock_max_missing_frames,
                lock_max_center_distance_ratio=config.roi_lock_max_center_distance_ratio,
                lock_strict=config.roi_lock_strict,
            )
        else:
            from .yolo_scissors_roi import (  # noqa: PLC0415
                SharedYoloScissorsROIProvider,
                YoloScissorsROIConfig,
            )

            roi_detector = SharedYoloScissorsROIProvider(
                video_path=path,
                config=YoloScissorsROIConfig(
                    roi_expand_x=config.roi_expand_x,
                    roi_expand_y=config.roi_expand_y,
                    roi_extra_down_ratio=config.roi_extra_down_ratio,
                    roi_extra_up_ratio=config.roi_extra_up_ratio,
                    max_roi_hold_frames=config.max_roi_hold_frames,
                    confidence_threshold=config.yolo_confidence_threshold,
                    roi_smoothing_enabled=config.roi_smoothing_enabled,
                    roi_smoothing_alpha=config.roi_smoothing_alpha,
                    artifact_path=config.yolo_scissors_artifact_path,
                ),
            )

    print("[OF] feature extraction started", flush=True)
    try:
        while True:
            success, curr_frame = cap.read()
            if not success or curr_frame is None:
                break

            curr_frame = _resize_frame_if_needed(
                curr_frame,
                config.resize_width,
                config.resize_height,
            )
            curr_gray = _to_gray(curr_frame)

            prev_for_flow = _blur_gray_for_flow(prev_gray, config.gaussian_blur_kernel)
            curr_for_flow = _blur_gray_for_flow(curr_gray, config.gaussian_blur_kernel)
            roi_used = False
            roi = None
            yolo_debug: dict | None = None
            if roi_detector is not None:
                if active_roi_source == "yolo_scissors_expanded":
                    yolo_roi_frames += 1
                    roi = roi_detector.detect(curr_frame, frame_index)
                    yolo_debug = roi_detector.last_debug_dict()
                else:
                    roi = roi_detector.detect(curr_frame)

            if roi is not None:
                if crop_to_roi is None or embed_roi_flow_in_canvas is None:
                    raise RuntimeError("ROI helpers were not initialized.")
                roi_prev = crop_to_roi(prev_for_flow, roi)
                roi_curr = crop_to_roi(curr_for_flow, roi)
                roi_flow = cv2.calcOpticalFlowFarneback(
                    prev=roi_prev,
                    next=roi_curr,
                    flow=None,
                    pyr_scale=config.pyr_scale,
                    levels=config.levels,
                    winsize=config.winsize,
                    iterations=config.iterations,
                    poly_n=config.poly_n,
                    poly_sigma=config.poly_sigma,
                    flags=config.flags,
                )
                height, width = curr_gray.shape[:2]
                flow = embed_roi_flow_in_canvas(
                    roi_flow=roi_flow,
                    roi=roi,
                    height=height,
                    width=width,
                )
                roi_used = True
                feature_flow = roi_flow if active_roi_source == "yolo_scissors_expanded" else flow
            else:
                flow = cv2.calcOpticalFlowFarneback(
                    prev=prev_for_flow,
                    next=curr_for_flow,
                    flow=None,
                    pyr_scale=config.pyr_scale,
                    levels=config.levels,
                    winsize=config.winsize,
                    iterations=config.iterations,
                    poly_n=config.poly_n,
                    poly_sigma=config.poly_sigma,
                    flags=config.flags,
                )
                feature_flow = flow

            timestamp_sec = frame_index / fps
            fallback_used = bool(active_roi_source != "none" and roi is None)
            fallback_reason = "roi_not_found" if fallback_used else None
            original_scissors_bbox = None
            yolo_scissors_bbox = None
            expanded_roi_bbox = list(roi) if roi is not None else None
            expanded_optical_flow_roi = list(roi) if roi is not None else None
            expanded_roi_bbox_raw = None
            expanded_roi_bbox_smoothed = list(roi) if roi is not None else None
            detection_confidence = None
            roi_reused_from_previous = False
            frame_roi_source = active_roi_source if roi is not None else "none"
            if yolo_debug is not None:
                original_scissors_bbox = yolo_debug.get("original_scissors_bbox")
                yolo_scissors_bbox = original_scissors_bbox
                expanded_roi_bbox = yolo_debug.get("expanded_roi_bbox")
                expanded_optical_flow_roi = expanded_roi_bbox
                expanded_roi_bbox_raw = yolo_debug.get("expanded_roi_bbox_raw")
                expanded_roi_bbox_smoothed = yolo_debug.get(
                    "expanded_roi_bbox_smoothed"
                )
                detection_confidence = yolo_debug.get("detection_confidence")
                frame_roi_source = str(yolo_debug.get("roi_source") or frame_roi_source)
                roi_reused_from_previous = bool(
                    yolo_debug.get("roi_reused_from_previous", False)
                )
                fallback_used = bool(yolo_debug.get("fallback_used", fallback_used))
                fallback_reason = yolo_debug.get("fallback_reason") or fallback_reason
                if fallback_used and roi is not None:
                    print(
                        f"[OF] YOLO missing frame={frame_index} using previous ROI",
                        flush=True,
                    )
                elif fallback_used:
                    print(
                        f"[OF] YOLO failed frame={frame_index} reason={fallback_reason}",
                        flush=True,
                    )
                elif yolo_roi_frames % 25 == 0:
                    print(
                        "[OF] YOLO detected scissors "
                        f"frame={frame_index} conf={detection_confidence}",
                        flush=True,
                    )

            features = extract_frame_flow_features(
                flow=feature_flow,
                frame_index=frame_index,
                timestamp_sec=timestamp_sec,
                motion_threshold=config.motion_threshold,
                roi_used=roi_used,
                roi_source=frame_roi_source,  # type: ignore[arg-type]
                roi_found=roi is not None,
                original_scissors_bbox=original_scissors_bbox,
                yolo_scissors_bbox=yolo_scissors_bbox,
                expanded_roi_bbox=expanded_roi_bbox,
                expanded_optical_flow_roi=expanded_optical_flow_roi,
                expanded_roi_bbox_raw=expanded_roi_bbox_raw,
                expanded_roi_bbox_smoothed=expanded_roi_bbox_smoothed,
                detection_confidence=detection_confidence,
                roi_reused_from_previous=roi_reused_from_previous,
                fallback_used=fallback_used,
                fallback_reason=fallback_reason,
                roi_area_ratio=_roi_area_ratio(roi, curr_gray.shape),
            )
            if frame_features:
                features.vibration_delta = round(
                    abs(features.mean_magnitude - frame_features[-1].mean_magnitude),
                    6,
                )
            else:
                features.vibration_delta = 0.0
            frame_features.append(features)
            if yolo_debug is not None:
                print(
                    "[OF][YOLO-ROI] "
                    f"frame={frame_index} "
                    f"detected={not roi_reused_from_previous and original_scissors_bbox is not None} "
                    f"bbox={original_scissors_bbox} "
                    f"expanded_roi={expanded_optical_flow_roi} "
                    f"mean_mag={features.mean_magnitude:.6f} "
                    f"max_mag={features.max_magnitude:.6f} "
                    f"vibration={features.vibration_delta:.6f}",
                    flush=True,
                )
            processed_count += 1

            if processed_count % 25 == 0:
                elapsed = time.perf_counter() - extraction_start
                speed = processed_count / max(elapsed, 1e-6)
                print(
                    f"[OF] processed frame {frame_index}/{metadata.frame_count} | "
                    f"processed={processed_count} | "
                    f"yolo_roi_frames={yolo_roi_frames} | "
                    f"elapsed={elapsed:.1f}s | "
                    f"speed={speed:.2f} fps",
                    flush=True,
                )

            prev_gray = curr_gray
            frame_index += 1
    finally:
        cap.release()
        if roi_detector is not None:
            roi_detector.close()

    feature_extraction_time_sec = time.perf_counter() - extraction_start
    print("[OF] feature extraction finished", flush=True)
    print(f"[OF] total_processed_frames={processed_count}", flush=True)
    print(f"[OF] total_yolo_roi_frames={yolo_roi_frames}", flush=True)
    print(
        f"[OF] feature_extraction_time_sec={feature_extraction_time_sec:.2f}",
        flush=True,
    )

    return metadata, frame_features
