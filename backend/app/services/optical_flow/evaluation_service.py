from __future__ import annotations

from dataclasses import dataclass

from .schemas import ComparisonMetrics, SummaryOpticalFlowResult


@dataclass
class OpticalFlowEvaluationConfig:
    magnitude_ref: float = 1.0
    motion_area_ref: float = 0.2
    angle_ref_deg: float = 90.0

    magnitude_weight: float = 0.4
    motion_area_weight: float = 0.2
    angle_weight: float = 0.4


def _clamp_01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def compute_magnitude_similarity(
    magnitude_curve_mae: float,
    reference_value: float,
) -> float:
    if reference_value <= 0:
        raise ValueError("reference_value must be > 0")
    return _clamp_01(1.0 - (magnitude_curve_mae / reference_value))


def compute_motion_area_similarity(
    motion_area_curve_mae: float,
    reference_value: float,
) -> float:
    if reference_value <= 0:
        raise ValueError("reference_value must be > 0")
    return _clamp_01(1.0 - (motion_area_curve_mae / reference_value))


def compute_angle_similarity(
    angle_curve_mae_deg: float,
    reference_value_deg: float,
) -> float:
    if reference_value_deg <= 0:
        raise ValueError("reference_value_deg must be > 0")
    return _clamp_01(1.0 - (angle_curve_mae_deg / reference_value_deg))


def compute_optical_flow_score(
    magnitude_similarity: float,
    motion_area_similarity: float,
    angle_similarity: float,
    config: OpticalFlowEvaluationConfig | None = None,
) -> float:
    config = config or OpticalFlowEvaluationConfig()

    total_weight = (
        config.magnitude_weight
        + config.motion_area_weight
        + config.angle_weight
    )
    if total_weight <= 0:
        raise ValueError("Total weight must be > 0")

    weighted_score_01 = (
        config.magnitude_weight * magnitude_similarity
        + config.motion_area_weight * motion_area_similarity
        + config.angle_weight * angle_similarity
    ) / total_weight

    return round(weighted_score_01 * 100.0, 6)


def evaluate_optical_flow_metrics(
    metrics: ComparisonMetrics,
    config: OpticalFlowEvaluationConfig | None = None,
) -> dict:
    config = config or OpticalFlowEvaluationConfig()

    magnitude_similarity = compute_magnitude_similarity(
        magnitude_curve_mae=metrics.magnitude_curve_mae,
        reference_value=config.magnitude_ref,
    )
    motion_area_similarity = compute_motion_area_similarity(
        motion_area_curve_mae=metrics.motion_area_curve_mae,
        reference_value=config.motion_area_ref,
    )
    angle_similarity = compute_angle_similarity(
        angle_curve_mae_deg=metrics.angle_curve_mae_deg,
        reference_value_deg=config.angle_ref_deg,
    )

    optical_flow_score = compute_optical_flow_score(
        magnitude_similarity=magnitude_similarity,
        motion_area_similarity=motion_area_similarity,
        angle_similarity=angle_similarity,
        config=config,
    )

    return {
        "similarities": {
            "magnitude_similarity": round(magnitude_similarity, 6),
            "motion_area_similarity": round(motion_area_similarity, 6),
            "angle_similarity": round(angle_similarity, 6),
        },
        "score": {
            "optical_flow_score": optical_flow_score
        },
        "config_used": {
            "magnitude_ref": config.magnitude_ref,
            "motion_area_ref": config.motion_area_ref,
            "angle_ref_deg": config.angle_ref_deg,
            "magnitude_weight": config.magnitude_weight,
            "motion_area_weight": config.motion_area_weight,
            "angle_weight": config.angle_weight,
        },
    }


def evaluate_optical_flow_summary(
    summary_result: SummaryOpticalFlowResult,
    config: OpticalFlowEvaluationConfig | None = None,
) -> dict:
    return evaluate_optical_flow_metrics(
        metrics=summary_result.comparison_metrics,
        config=config,
    )
