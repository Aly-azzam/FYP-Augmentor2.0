from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List

import cv2
import numpy as np

from .feature_extractor import extract_frame_flow_features
from .schemas import FrameFlowFeatures, VideoMetadata


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

    if config.use_hand_roi:
        from .hand_roi import (
            HandROIDetector,
            crop_to_roi,
            embed_roi_flow_in_canvas,
        )

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
            roi = roi_detector.detect(curr_frame) if roi_detector is not None else None

            if roi is not None:
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

            timestamp_sec = frame_index / fps
            features = extract_frame_flow_features(
                flow=flow,
                frame_index=frame_index,
                timestamp_sec=timestamp_sec,
                motion_threshold=config.motion_threshold,
                roi_used=roi_used,
            )
            frame_features.append(features)

            prev_gray = curr_gray
            frame_index += 1
    finally:
        cap.release()
        if roi_detector is not None:
            roi_detector.close()

    return metadata, frame_features
