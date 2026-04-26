from __future__ import annotations

from typing import List

import cv2
import numpy as np

from .schemas import FrameFlowFeatures, VideoFlowSummary


def compute_magnitude_and_angle(flow: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Convert dense optical flow vectors into magnitude and angle matrices.

    Args:
        flow: Optical flow array of shape (H, W, 2)

    Returns:
        magnitude: Motion magnitude per pixel
        angle_deg: Motion angle per pixel in degrees [0, 360)
    """
    if flow.ndim != 3 or flow.shape[2] != 2:
        raise ValueError("Flow must have shape (H, W, 2).")

    flow_x = flow[..., 0]
    flow_y = flow[..., 1]
    magnitude, angle_rad = cv2.cartToPolar(flow_x, flow_y, angleInDegrees=False)
    angle_deg = np.degrees(angle_rad) % 360.0
    return magnitude, angle_deg


def compute_motion_area_ratio(
    magnitude: np.ndarray,
    motion_threshold: float = 2.0,
) -> float:
    """
    Estimate how much of the frame is considered 'moving'.

    Args:
        magnitude: Magnitude matrix
        motion_threshold: Pixels with magnitude > threshold are considered motion

    Returns:
        Ratio in [0, 1]
    """
    if magnitude.size == 0:
        return 0.0

    moving_mask = magnitude > motion_threshold
    return float(np.mean(moving_mask))


def compute_mean_angle_deg(
    angle_deg: np.ndarray,
    magnitude: np.ndarray | None = None,
) -> float:
    """
    Compute average angle in degrees.

    If magnitude is provided, use a weighted circular mean so stronger motion
    contributes more than weak/noisy motion.

    Returns:
        Angle in [0, 360)
    """
    if angle_deg.size == 0:
        return 0.0

    angles_rad = np.radians(angle_deg)

    if magnitude is None:
        weights = np.ones_like(angle_deg, dtype=np.float32)
    else:
        weights = np.asarray(magnitude, dtype=np.float32)

    weight_sum = float(np.sum(weights))
    if weight_sum <= 1e-8:
        return 0.0

    sin_sum = float(np.sum(np.sin(angles_rad) * weights))
    cos_sum = float(np.sum(np.cos(angles_rad) * weights))

    mean_angle_rad = np.arctan2(sin_sum, cos_sum)
    mean_angle_deg = np.degrees(mean_angle_rad) % 360.0
    return float(mean_angle_deg)


def extract_frame_flow_features(
    flow: np.ndarray,
    frame_index: int,
    timestamp_sec: float,
    motion_threshold: float = 2.0,
) -> FrameFlowFeatures:
    """
    Extract summary features from one dense optical flow frame.
    """
    magnitude, angle_deg = compute_magnitude_and_angle(flow)

    mean_magnitude = float(np.mean(magnitude)) if magnitude.size > 0 else 0.0
    max_magnitude = float(np.max(magnitude)) if magnitude.size > 0 else 0.0
    mean_angle = compute_mean_angle_deg(angle_deg, magnitude=magnitude)
    motion_area_ratio = compute_motion_area_ratio(
        magnitude,
        motion_threshold=motion_threshold,
    )

    return FrameFlowFeatures(
        frame_index=frame_index,
        timestamp_sec=timestamp_sec,
        mean_magnitude=round(mean_magnitude, 6),
        max_magnitude=round(max_magnitude, 6),
        mean_angle_deg=round(mean_angle, 6),
        motion_area_ratio=round(motion_area_ratio, 6),
    )


def _safe_mean(values: List[float]) -> float:
    return float(np.mean(values)) if values else 0.0


def smooth_signal(values: List[float], window_size: int = 5) -> List[float]:
    """
    Smooth a 1D signal with a simple moving average.

    Args:
        values: Input list of numeric values
        window_size: Smoothing window size

    Returns:
        Smoothed list with the same length as input
    """
    if window_size <= 1 or len(values) < window_size:
        return values

    smoothed: List[float] = []
    half = window_size // 2

    for i in range(len(values)):
        start = max(0, i - half)
        end = min(len(values), i + half + 1)
        smoothed.append(float(np.mean(values[start:end])))

    return smoothed


def _compute_motion_stability_score(mean_magnitudes: List[float]) -> float:
    """
    Simple stability score in [0, 1].

    Idea:
    - lower variation in motion magnitude => more stable
    - higher variation => less stable

    This is a simple first-version heuristic, not a final scientific metric.
    """
    if not mean_magnitudes:
        return 0.0

    arr = np.asarray(mean_magnitudes, dtype=np.float32)
    avg = float(np.mean(arr))

    if avg <= 1e-8:
        return 1.0

    std = float(np.std(arr))
    coeff_var = std / avg

    stability = 1.0 / (1.0 + coeff_var)
    stability = float(np.clip(stability, 0.0, 1.0))
    return stability


def _compute_magnitude_std(mean_magnitudes: List[float]) -> float:
    """
    Standard deviation of per-frame mean magnitudes.
    """
    if not mean_magnitudes:
        return 0.0

    arr = np.asarray(mean_magnitudes, dtype=np.float32)
    return float(np.std(arr))


def _compute_magnitude_jitter(mean_magnitudes: List[float]) -> float:
    """
    Average absolute frame-to-frame change in mean magnitude.
    """
    if len(mean_magnitudes) < 2:
        return 0.0

    arr = np.asarray(mean_magnitudes, dtype=np.float32)
    diffs = np.abs(np.diff(arr))
    return float(np.mean(diffs)) if diffs.size > 0 else 0.0


def _compute_vibration_score(magnitude_jitter: float) -> float:
    """
    Simple normalized vibration score in [0, 1].
    """
    return float(np.clip(magnitude_jitter / 1.0, 0.0, 1.0))


def _compute_robust_peak_magnitude(max_magnitudes: List[float]) -> float:
    """
    Robust peak based on the 95th percentile instead of raw max.
    This reduces the effect of extreme outlier frames.
    """
    if not max_magnitudes:
        return 0.0

    arr = np.asarray(max_magnitudes, dtype=np.float32)
    return float(np.percentile(arr, 95))


def build_video_flow_summary(
    frame_features: List[FrameFlowFeatures],
) -> VideoFlowSummary:
    """
    Aggregate per-frame flow features into a video-level summary.
    """
    if not frame_features:
        return VideoFlowSummary(
            avg_magnitude=0.0,
            peak_magnitude=0.0,
            avg_motion_area_ratio=0.0,
            avg_angle_deg=0.0,
            motion_stability_score=0.0,
            magnitude_std=0.0,
            magnitude_jitter=0.0,
            vibration_score=0.0,
        )

    raw_mean_magnitudes = [f.mean_magnitude for f in frame_features]
    mean_magnitudes = smooth_signal(
        raw_mean_magnitudes,
        window_size=5,
    )

    max_magnitudes = [f.max_magnitude for f in frame_features]

    motion_area_ratios = smooth_signal(
        [f.motion_area_ratio for f in frame_features],
        window_size=5,
    )

    # Keep angles unsmoothed here because angles are circular values
    mean_angles = [f.mean_angle_deg for f in frame_features]

    avg_magnitude = _safe_mean(mean_magnitudes)
    peak_magnitude = _compute_robust_peak_magnitude(max_magnitudes)
    avg_motion_area_ratio = _safe_mean(motion_area_ratios)
    avg_angle_deg = compute_mean_angle_deg(
        np.asarray(mean_angles, dtype=np.float32),
        magnitude=np.asarray(mean_magnitudes, dtype=np.float32),
    )
    motion_stability_score = _compute_motion_stability_score(mean_magnitudes)
    magnitude_std = _compute_magnitude_std(raw_mean_magnitudes)
    magnitude_jitter = _compute_magnitude_jitter(raw_mean_magnitudes)
    vibration_score = _compute_vibration_score(magnitude_jitter)

    return VideoFlowSummary(
        avg_magnitude=round(avg_magnitude, 6),
        peak_magnitude=round(peak_magnitude, 6),
        avg_motion_area_ratio=round(avg_motion_area_ratio, 6),
        avg_angle_deg=round(avg_angle_deg, 6),
        motion_stability_score=round(motion_stability_score, 6),
        magnitude_std=round(magnitude_std, 6),
        magnitude_jitter=round(magnitude_jitter, 6),
        vibration_score=round(vibration_score, 6),
    )
