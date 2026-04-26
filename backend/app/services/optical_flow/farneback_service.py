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
        )
        frame_features.append(features)

        prev_gray = curr_gray
        frame_index += 1

    cap.release()
    return metadata, frame_features
