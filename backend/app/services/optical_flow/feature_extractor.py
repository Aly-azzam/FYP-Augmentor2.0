from __future__ import annotations

from typing import List, Literal

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
    roi_used: bool = False,
    roi_source: Literal[
        "none",
        "mediapipe_hand",
        "yolo",
        "yolo_scissors",
        "yolo_scissors_expanded",
        "previous_yolo_bbox",
        "previous_yolo_fallback",
        "full_frame_fallback",
    ] = "none",
    roi_found: bool = False,
    original_scissors_bbox: list[float] | None = None,
    yolo_scissors_bbox: list[float] | None = None,
    expanded_roi_bbox: list[int] | None = None,
    expanded_optical_flow_roi: list[int] | None = None,
    expanded_roi_bbox_raw: list[int] | None = None,
    expanded_roi_bbox_smoothed: list[int] | None = None,
    detection_confidence: float | None = None,
    roi_reused_from_previous: bool = False,
    fallback_used: bool = False,
    fallback_reason: str | None = None,
    roi_area_ratio: float = 0.0,
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

    effective_yolo_bbox = yolo_scissors_bbox or original_scissors_bbox
    effective_roi = expanded_optical_flow_roi or expanded_roi_bbox
    roi_width = max(0, int(effective_roi[2]) - int(effective_roi[0])) if effective_roi else 0
    roi_height = max(0, int(effective_roi[3]) - int(effective_roi[1])) if effective_roi else 0
    yolo_detected = bool(
        roi_source in {"yolo", "yolo_scissors", "yolo_scissors_expanded"}
        and roi_found
        and not roi_reused_from_previous
        and effective_yolo_bbox is not None
    )

    return FrameFlowFeatures(
        frame_index=frame_index,
        timestamp_sec=timestamp_sec,
        mean_magnitude=round(mean_magnitude, 6),
        max_magnitude=round(max_magnitude, 6),
        mean_angle_deg=round(mean_angle, 6),
        motion_area_ratio=round(motion_area_ratio, 6),
        roi_used=roi_used,
        roi_source=roi_source,
        roi_found=roi_found,
        original_scissors_bbox=original_scissors_bbox,
        yolo_scissors_bbox=effective_yolo_bbox,
        expanded_roi_bbox=expanded_roi_bbox,
        expanded_optical_flow_roi=effective_roi,
        expanded_roi_bbox_raw=expanded_roi_bbox_raw,
        expanded_roi_bbox_smoothed=expanded_roi_bbox_smoothed,
        detection_confidence=detection_confidence,
        yolo_detected=yolo_detected,
        yolo_confidence=detection_confidence if yolo_detected else None,
        scissors_bbox_xyxy=effective_yolo_bbox if yolo_detected else None,
        expanded_roi_xyxy=effective_roi,
        roi_width=roi_width,
        roi_height=roi_height,
        flow_mean_magnitude=round(mean_magnitude, 6),
        flow_max_magnitude=round(max_magnitude, 6),
        flow_motion_area_ratio=round(motion_area_ratio, 6),
        roi_reused_from_previous=roi_reused_from_previous,
        fallback_used=fallback_used,
        fallback_reason=fallback_reason,
        roi_area_ratio=round(float(roi_area_ratio), 6),
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


def _compute_high_frequency_vibration(
    raw_signal: List[float],
    smoothed_signal: List[float],
) -> tuple[float, float]:
    """
    Measure rapid motion instability as the residual from a smoothed signal.
    """
    if not raw_signal or len(raw_signal) != len(smoothed_signal):
        return 0.0, 0.0

    raw_arr = np.asarray(raw_signal, dtype=np.float32)
    smoothed_arr = np.asarray(smoothed_signal, dtype=np.float32)
    high_freq = np.abs(raw_arr - smoothed_arr)

    high_freq_mean = float(np.mean(high_freq)) if high_freq.size > 0 else 0.0
    high_freq_max = float(np.max(high_freq)) if high_freq.size > 0 else 0.0
    return high_freq_mean, high_freq_max


def _compute_vibration_score(vibration_high_freq_mean: float) -> float:
    """
    Normalize high-frequency motion instability into [0, 1].
    """
    return min(1.0, vibration_high_freq_mean / 0.2)


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
    roi_enabled: bool = False,
    roi_source_used: Literal[
        "none",
        "mediapipe_hand",
        "yolo",
        "yolo_scissors",
        "yolo_scissors_expanded",
        "previous_yolo_bbox",
        "previous_yolo_fallback",
        "full_frame_fallback",
    ] = "none",
    roi_smoothing_enabled: bool = False,
    roi_smoothing_alpha: float | None = None,
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
            vibration_high_freq_mean=0.0,
            vibration_high_freq_max=0.0,
            vibration_score=0.0,
            roi_enabled=roi_enabled,
            roi_source_used=roi_source_used if roi_enabled else "none",
            total_frames=0,
            processed_pairs=0,
            yolo_detection_count=0,
            fallback_count=0,
            full_frame_fallback_count=0,
            average_flow_mean_magnitude=0.0,
            peak_flow_magnitude=0.0,
            roi_smoothing_enabled=roi_smoothing_enabled,
            roi_smoothing_alpha=roi_smoothing_alpha,
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
    vibration_high_freq_mean, vibration_high_freq_max = _compute_high_frequency_vibration(
        raw_signal=raw_mean_magnitudes,
        smoothed_signal=mean_magnitudes,
    )
    vibration_score = _compute_vibration_score(vibration_high_freq_mean)
    roi_frames_used = sum(1 for f in frame_features if f.roi_used)
    roi_enabled = roi_enabled or roi_frames_used > 0
    roi_fallback_frames = len(frame_features) - roi_frames_used if roi_enabled else 0
    roi_usage_ratio = (
        roi_frames_used / len(frame_features)
        if frame_features and roi_enabled
        else 0.0
    )
    if roi_source_used == "none" and roi_enabled:
        if any(f.roi_source == "yolo_scissors_expanded" for f in frame_features):
            roi_source_used = "yolo_scissors_expanded"
        elif any(f.roi_source == "yolo_scissors" for f in frame_features):
            roi_source_used = "yolo_scissors"
        elif any(f.roi_source == "mediapipe_hand" for f in frame_features):
            roi_source_used = "mediapipe_hand"

    yolo_detection_frames = sum(
        1
        for f in frame_features
        if f.roi_source in {"yolo", "yolo_scissors", "yolo_scissors_expanded"}
        and f.roi_found
        and not f.roi_reused_from_previous
    )
    yolo_detection_ratio = (
        yolo_detection_frames / len(frame_features)
        if frame_features and roi_source_used in {"yolo_scissors", "yolo_scissors_expanded"}
        else 0.0
    )
    reused_roi_frame_count = sum(1 for f in frame_features if f.roi_reused_from_previous)
    fallback_frame_count = sum(1 for f in frame_features if f.fallback_used)
    yolo_fallback_count = sum(
        1
        for f in frame_features
        if f.roi_source in {"previous_yolo_fallback", "full_frame_fallback"}
    )
    full_frame_fallback_count = sum(
        1 for f in frame_features if f.roi_source == "full_frame_fallback"
    )
    roi_area_ratios = [f.roi_area_ratio for f in frame_features if f.roi_used]
    average_roi_area_ratio = _safe_mean(roi_area_ratios)

    return VideoFlowSummary(
        avg_magnitude=round(avg_magnitude, 6),
        peak_magnitude=round(peak_magnitude, 6),
        avg_motion_area_ratio=round(avg_motion_area_ratio, 6),
        avg_angle_deg=round(avg_angle_deg, 6),
        motion_stability_score=round(motion_stability_score, 6),
        magnitude_std=round(magnitude_std, 6),
        magnitude_jitter=round(magnitude_jitter, 6),
        vibration_high_freq_mean=round(vibration_high_freq_mean, 6),
        vibration_high_freq_max=round(vibration_high_freq_max, 6),
        vibration_score=round(vibration_score, 6),
        roi_enabled=roi_enabled,
        roi_frames_used=roi_frames_used,
        roi_fallback_frames=roi_fallback_frames,
        roi_usage_ratio=round(roi_usage_ratio, 6),
        roi_source_used=roi_source_used if roi_enabled else "none",
        total_frames=len(frame_features) + 1 if frame_features else 0,
        processed_pairs=len(frame_features),
        yolo_detection_count=yolo_detection_frames,
        yolo_detection_ratio=round(yolo_detection_ratio, 6),
        fallback_count=yolo_fallback_count,
        full_frame_fallback_count=full_frame_fallback_count,
        average_flow_mean_magnitude=round(avg_magnitude, 6),
        peak_flow_magnitude=round(max(max_magnitudes), 6) if max_magnitudes else 0.0,
        reused_roi_frame_count=reused_roi_frame_count,
        fallback_frame_count=fallback_frame_count,
        average_roi_area_ratio=round(average_roi_area_ratio, 6),
        roi_smoothing_enabled=roi_smoothing_enabled,
        roi_smoothing_alpha=roi_smoothing_alpha,
    )
