"""Orchestrate the full Evaluation Engine pipeline."""

from __future__ import annotations

import uuid
from pathlib import Path
from typing import Any

from app.schemas.evaluation_schema import EvaluationResult, MetricSet
from app.services.angle_metrics_service import compute_angle_deviation
from app.services.comparison_service import (
    load_motion_data,
    pair_motion_data,
)
from app.services.feedback_structuring_service import build_summary, build_vlm_payload
from app.services.scoring_service import compute_final_score
from app.services.tool_metrics_service import compute_tool_alignment_deviation
from app.services.trajectory_metrics_service import compute_trajectory_deviation
from app.services.velocity_metrics_service import compute_velocity_difference

MotionSource = str | Path | dict[str, Any]


def generate_evaluation_id() -> str:
    """Generate a simple evaluation identifier."""
    return f"eval_{uuid.uuid4().hex[:8]}"


def build_metric_set(paired_motion_data: dict[str, Any]) -> MetricSet:
    """Compute all evaluation metrics from paired motion data."""
    return MetricSet(
        angle_deviation=compute_angle_deviation(paired_motion_data),
        trajectory_deviation=compute_trajectory_deviation(paired_motion_data),
        velocity_difference=compute_velocity_difference(paired_motion_data),
        tool_alignment_deviation=compute_tool_alignment_deviation(paired_motion_data),
    )


def evaluate_motion_pair(
    expert_motion_source: MotionSource,
    learner_motion_source: MotionSource,
) -> EvaluationResult:
    """Run the complete evaluation flow from motion inputs to final result."""
    expert_motion_data = load_motion_data(expert_motion_source)
    learner_motion_data = load_motion_data(learner_motion_source)

    paired_motion_data = pair_motion_data(
        expert_data=expert_motion_data,
        learner_data=learner_motion_data,
    )

    metrics = build_metric_set(paired_motion_data)
    score_result = compute_final_score(metrics.model_dump())
    final_score = score_result["score"]

    summary = build_summary(score=final_score, metrics=metrics)
    vlm_payload = build_vlm_payload(
        score=final_score,
        metrics=metrics,
        summary=summary,
    )

    return EvaluationResult(
        evaluation_id=generate_evaluation_id(),
        score=final_score,
        metrics=metrics,
        summary=summary,
        vlm_payload=vlm_payload,
    )


async def run_evaluation_pipeline(
    expert_motion_source: MotionSource,
    learner_motion_source: MotionSource,
) -> EvaluationResult:
    """Async wrapper around the evaluation flow for later API integration."""
    return evaluate_motion_pair(
        expert_motion_source=expert_motion_source,
        learner_motion_source=learner_motion_source,
    )
