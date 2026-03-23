"""Evaluation engine orchestration for both internal and paired pipelines."""

from __future__ import annotations

import uuid
from pathlib import Path
from typing import Any

from app.schemas.evaluation_schema import EvaluationResult, MetricSet
from app.services.angle_metrics_service import (
    compute_angle_deviation,
    compute_video_hand_angles,
)
from app.services.comparison_service import load_motion_data, pair_motion_data
from app.services.feedback_structuring_service import build_summary, build_vlm_payload
from app.services.perception_service import run_perception_pipeline
from app.services.scoring_service import compute_final_score, compute_internal_motion_score
from app.services.tool_metrics_service import compute_tool_alignment_deviation
from app.services.trajectory_metrics_service import (
    compute_trajectory_deviation,
    compute_video_trajectory_metrics,
)
from app.services.velocity_metrics_service import (
    compute_velocity_difference,
    compute_video_velocity_metrics,
)

MotionSource = str | Path | dict[str, Any]


def validate_perception_output(perception_output: Any) -> list[dict]:
    """Validate perception output structure and return the frames list."""
    if isinstance(perception_output, dict):
        required_keys = {"video_path", "total_frames", "processed_frames", "frames"}
        missing = required_keys - set(perception_output.keys())
        if missing:
            missing_str = ", ".join(sorted(missing))
            raise ValueError(f"validate_perception_output: missing keys: {missing_str}.")

        frames = perception_output["frames"]
    else:
        if not hasattr(perception_output, "frames"):
            raise ValueError("validate_perception_output: perception_output must expose frames.")
        frames = perception_output.frames

    if not isinstance(frames, list):
        raise ValueError("validate_perception_output: 'frames' must be a list.")

    return frames


def run_full_evaluation(video_path: str) -> dict[str, Any]:
    """Run the internal single-video evaluation pipeline."""
    if not isinstance(video_path, str) or not video_path.strip():
        raise ValueError("run_full_evaluation: video_path must be a non-empty string.")

    perception_output = run_perception_pipeline(video_path)
    frames = validate_perception_output(perception_output)

    angle_frames = compute_video_hand_angles(frames)
    trajectory_metrics = compute_video_trajectory_metrics(frames)
    velocity_metrics = compute_video_velocity_metrics(frames)
    scoring_output = compute_internal_motion_score(
        angle_frames,
        trajectory_metrics,
        velocity_metrics,
    )

    return {
        "video_path": video_path,
        "perception": perception_output,
        "angle_metrics": angle_frames,
        "trajectory_metrics": trajectory_metrics,
        "velocity_metrics": velocity_metrics,
        "scoring": scoring_output,
    }


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
    """Run the complete paired evaluation flow from motion inputs to final result."""
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
    """Async wrapper around paired evaluation flow for API integration."""
    return evaluate_motion_pair(
        expert_motion_source=expert_motion_source,
        learner_motion_source=learner_motion_source,
    )
