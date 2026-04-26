from __future__ import annotations

import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import List

import numpy as np

from .farneback_service import FarnebackConfig, compute_video_optical_flow_features
from .feature_extractor import (
    build_video_flow_summary,
    compute_mean_angle_deg,
    smooth_signal,
)
from .schemas import (
    ComparisonInfo,
    ComparisonMetrics,
    FrameFlowFeatures,
    InterpretationReady,
    OpticalFlowEvaluationResult,
    RawOpticalFlowResult,
    RunInfo,
    SummaryOpticalFlowResult,
)


def _truncate_to_shorter(
    expert_frames: List[FrameFlowFeatures],
    learner_frames: List[FrameFlowFeatures],
) -> tuple[List[FrameFlowFeatures], List[FrameFlowFeatures]]:
    frame_count_used = min(len(expert_frames), len(learner_frames))
    return expert_frames[:frame_count_used], learner_frames[:frame_count_used]


def _safe_mean(values: List[float]) -> float:
    return float(np.mean(values)) if values else 0.0


def _circular_angle_difference_deg(angle_a: float, angle_b: float) -> float:
    """
    Smallest absolute circular difference between two angles in degrees.
    Result is in [0, 180].
    """
    diff = abs(angle_a - angle_b) % 360.0
    return min(diff, 360.0 - diff)


def _compute_curve_mae(values_a: List[float], values_b: List[float]) -> float:
    if not values_a or not values_b:
        return 0.0

    arr_a = np.asarray(values_a, dtype=np.float32)
    arr_b = np.asarray(values_b, dtype=np.float32)
    return float(np.mean(np.abs(arr_a - arr_b)))


def _resample_signal(values: List[float], target_length: int = 100) -> List[float]:
    """
    Resample a 1D signal to a fixed target length using linear interpolation.
    """
    if not values:
        return []
    if len(values) == target_length:
        return values
    if len(values) == 1:
        return [values[0]] * target_length

    original_x = np.linspace(0.0, 1.0, num=len(values))
    target_x = np.linspace(0.0, 1.0, num=target_length)
    resampled = np.interp(target_x, original_x, values)
    return [float(v) for v in resampled]


def _compute_angle_curve_mae(
    expert_frames: List[FrameFlowFeatures],
    learner_frames: List[FrameFlowFeatures],
    active_motion_threshold: float = 0.1,
) -> float:
    """
    Compute angle MAE only on active-motion frames.

    Angle can be noisy when motion is tiny, so we ignore frames where both
    expert and learner have very weak motion.
    """
    if not expert_frames or not learner_frames:
        return 0.0

    diffs: list[float] = []

    for expert_frame, learner_frame in zip(expert_frames, learner_frames):
        expert_mag = expert_frame.mean_magnitude
        learner_mag = learner_frame.mean_magnitude

        if expert_mag < active_motion_threshold and learner_mag < active_motion_threshold:
            continue

        diff = _circular_angle_difference_deg(
            expert_frame.mean_angle_deg,
            learner_frame.mean_angle_deg,
        )
        diffs.append(diff)

    if not diffs:
        return 0.0

    return _safe_mean(diffs)


def _compute_global_direction_deg(frames: List[FrameFlowFeatures]) -> float:
    """
    Compute a proper circular mean of frame directions, weighted by frame mean magnitude.
    """
    if not frames:
        return 0.0

    angles = np.asarray([f.mean_angle_deg for f in frames], dtype=np.float32)
    magnitudes = np.asarray([f.mean_magnitude for f in frames], dtype=np.float32)
    return float(compute_mean_angle_deg(angles, magnitude=magnitudes))


def _compute_robust_peak_magnitude(frames: List[FrameFlowFeatures]) -> float:
    """
    More stable than raw max(): use the 95th percentile of per-frame max magnitude.
    """
    if not frames:
        return 0.0

    values = np.asarray([f.max_magnitude for f in frames], dtype=np.float32)
    return float(np.percentile(values, 95))


def _build_comparison_metrics(
    expert_frames: List[FrameFlowFeatures],
    learner_frames: List[FrameFlowFeatures],
) -> ComparisonMetrics:
    # Smooth curves first
    expert_mean_magnitudes = smooth_signal(
        [f.mean_magnitude for f in expert_frames],
        window_size=5,
    )
    learner_mean_magnitudes = smooth_signal(
        [f.mean_magnitude for f in learner_frames],
        window_size=5,
    )

    expert_motion_areas = smooth_signal(
        [f.motion_area_ratio for f in expert_frames],
        window_size=5,
    )
    learner_motion_areas = smooth_signal(
        [f.motion_area_ratio for f in learner_frames],
        window_size=5,
    )

    # Normalize both curves to same sample length
    expert_mean_magnitudes = _resample_signal(expert_mean_magnitudes, target_length=100)
    learner_mean_magnitudes = _resample_signal(learner_mean_magnitudes, target_length=100)

    expert_motion_areas = _resample_signal(expert_motion_areas, target_length=100)
    learner_motion_areas = _resample_signal(learner_motion_areas, target_length=100)

    expert_global_direction = _compute_global_direction_deg(expert_frames)
    learner_global_direction = _compute_global_direction_deg(learner_frames)

    expert_robust_peak = _compute_robust_peak_magnitude(expert_frames)
    learner_robust_peak = _compute_robust_peak_magnitude(learner_frames)

    mean_magnitude_difference = abs(
        _safe_mean(expert_mean_magnitudes) - _safe_mean(learner_mean_magnitudes)
    )
    peak_magnitude_difference = abs(expert_robust_peak - learner_robust_peak)
    motion_area_difference = abs(
        _safe_mean(expert_motion_areas) - _safe_mean(learner_motion_areas)
    )
    mean_direction_difference_deg = _circular_angle_difference_deg(
        expert_global_direction,
        learner_global_direction,
    )

    magnitude_curve_mae = _compute_curve_mae(
        expert_mean_magnitudes,
        learner_mean_magnitudes,
    )
    motion_area_curve_mae = _compute_curve_mae(
        expert_motion_areas,
        learner_motion_areas,
    )

    # Keep angle logic active-motion-aware, but do not smooth/resample angle directly yet
    angle_curve_mae_deg = _compute_angle_curve_mae(
        expert_frames,
        learner_frames,
        active_motion_threshold=0.1,
    )

    return ComparisonMetrics(
        mean_magnitude_difference=round(mean_magnitude_difference, 6),
        peak_magnitude_difference=round(peak_magnitude_difference, 6),
        motion_area_difference=round(motion_area_difference, 6),
        mean_direction_difference_deg=round(mean_direction_difference_deg, 6),
        magnitude_curve_mae=round(magnitude_curve_mae, 6),
        angle_curve_mae_deg=round(angle_curve_mae_deg, 6),
        motion_area_curve_mae=round(motion_area_curve_mae, 6),
    )


def _build_interpretation_ready(
    metrics: ComparisonMetrics,
    expert_summary,
    learner_summary,
) -> InterpretationReady:
    signal_values = {
        "mean_magnitude_difference": metrics.mean_magnitude_difference,
        "peak_magnitude_difference": metrics.peak_magnitude_difference,
        "motion_area_difference": metrics.motion_area_difference,
        "mean_direction_difference_deg": metrics.mean_direction_difference_deg,
        "magnitude_curve_mae": metrics.magnitude_curve_mae,
        "angle_curve_mae_deg": metrics.angle_curve_mae_deg,
        "motion_area_curve_mae": metrics.motion_area_curve_mae,
    }

    strongest_signal = max(signal_values, key=signal_values.get)
    weakest_signal = min(signal_values, key=signal_values.get)

    observations: list[str] = []

    if metrics.mean_magnitude_difference >= 0.3 or metrics.magnitude_curve_mae >= 0.5:
        if learner_summary.avg_magnitude < expert_summary.avg_magnitude:
            observations.append("Learner motion is weaker than expert overall")
        elif learner_summary.avg_magnitude > expert_summary.avg_magnitude:
            observations.append("Learner motion is stronger than expert overall")
        else:
            observations.append("Learner motion differs from expert overall")

    if metrics.motion_area_difference >= 0.08 or metrics.motion_area_curve_mae >= 0.1:
        observations.append("Learner covers a different active motion area than expert")

    if metrics.mean_direction_difference_deg >= 25 or metrics.angle_curve_mae_deg >= 40:
        observations.append("Learner motion direction differs noticeably from expert")

    if not observations:
        if metrics.mean_magnitude_difference < 0.2 and metrics.motion_area_difference < 0.05:
            main_observation = (
                "Learner motion is relatively close to expert in intensity and active motion area."
            )
        else:
            main_observation = (
                "Learner shows moderate motion differences compared with expert."
            )
    elif len(observations) == 1:
        main_observation = observations[0] + "."
    elif len(observations) == 2:
        main_observation = observations[0] + ", and " + observations[1].lower() + "."
    else:
        main_observation = (
            observations[0]
            + ", "
            + observations[1].lower()
            + ", and "
            + observations[2].lower()
            + "."
        )

    return InterpretationReady(
        main_observation=main_observation,
        strongest_signal=strongest_signal,
        weakest_signal=weakest_signal,
    )


def run_optical_flow_comparison(
    expert_video_path: str | Path,
    learner_video_path: str | Path,
    config: FarnebackConfig | None = None,
    optical_flow_evaluation: OpticalFlowEvaluationResult | None = None,
) -> tuple[RawOpticalFlowResult, SummaryOpticalFlowResult]:
    """
    Run Farneback optical flow on expert and learner videos, then compare them.

    Returns:
        raw_result, summary_result
    """
    start_time = time.perf_counter()
    created_at = datetime.now(timezone.utc)
    run_id = f"of_{uuid.uuid4().hex[:12]}"

    expert_metadata, expert_frames = compute_video_optical_flow_features(
        video_path=expert_video_path,
        config=config,
    )
    learner_metadata, learner_frames = compute_video_optical_flow_features(
        video_path=learner_video_path,
        config=config,
    )

    expert_used, learner_used = _truncate_to_shorter(expert_frames, learner_frames)
    frame_count_used = min(len(expert_used), len(learner_used))

    expert_summary = build_video_flow_summary(expert_used)
    learner_summary = build_video_flow_summary(learner_used)
    comparison_metrics = _build_comparison_metrics(expert_used, learner_used)
    interpretation_ready = _build_interpretation_ready(
        comparison_metrics,
        expert_summary,
        learner_summary,
    )

    processing_time_sec = time.perf_counter() - start_time

    run_info = RunInfo(
        run_id=run_id,
        method="farneback",
        created_at=created_at,
        processing_time_sec=round(processing_time_sec, 6),
    )

    raw_result = RawOpticalFlowResult(
        run=run_info,
        expert_video=expert_metadata,
        learner_video=learner_metadata,
        expert_frames=expert_used,
        learner_frames=learner_used,
    )

    summary_result = SummaryOpticalFlowResult(
        run=run_info,
        comparison=ComparisonInfo(
            frame_count_used=frame_count_used,
            alignment_mode="truncate_to_shorter_video",
        ),
        expert_summary=expert_summary,
        learner_summary=learner_summary,
        comparison_metrics=comparison_metrics,
        interpretation_ready=interpretation_ready,
        optical_flow_evaluation=optical_flow_evaluation,
    )

    return raw_result, summary_result
