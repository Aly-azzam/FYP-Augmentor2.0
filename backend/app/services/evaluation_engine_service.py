"""Evaluation engine orchestration for both internal and paired pipelines."""

from __future__ import annotations

import logging
import uuid
from pathlib import Path
from typing import Any

from sqlalchemy import text

from app.schemas.evaluation_schema import EvaluationResult as EvaluationResultSchema, MetricSet
from app.models.attempt import Attempt
from app.models.evaluation import Evaluation as EvaluationModel
from app.models.video import Video
from app.core.database import SessionLocal, engine
from app.services.angle_metrics_service import (
    compute_angle_deviation,
    compute_hand_openness_deviation,
    compute_video_hand_angles,
)
from app.services.comparison_service import (
    detect_key_error_moments,
    load_motion_data,
    map_errors_to_phases,
    pair_motion_data,
    pair_motion_data_with_alignment,
)
from app.services.feedback_structuring_service import (
    build_explanation_payload,
    build_summary,
    build_vlm_payload,
    generate_explanation,
)
from app.core.config import settings
from app.core.evaluation_constants import DTW_SIMILARITY_DECAY
from app.services.perception_service import run_perception_pipeline
from app.services.scoring_service import compute_final_score, compute_internal_motion_score
from app.services.temporal_alignment_service import align_sequences
from app.services.tool_metrics_service import compute_tool_alignment_deviation
from app.services.trajectory_metrics_service import (
    compute_timing_score,
    compute_trajectory_deviation,
    compute_video_trajectory_metrics,
)
from app.services.velocity_metrics_service import (
    compute_smoothness_score,
    compute_velocity_difference,
    compute_video_velocity_metrics,
)
from app.services.vjepa_service import analyze_video_semantics

MotionSource = str | Path | dict[str, Any]
logger = logging.getLogger(__name__)


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


def compute_dtw_similarity(dtw_normalized_cost: float) -> float:
    """Convert DTW normalized cost to a similarity score in [0,1]."""
    return 1.0 / (1.0 + DTW_SIMILARITY_DECAY * max(0.0, dtw_normalized_cost))


def build_metric_set(
    paired_motion_data: dict[str, Any],
    dtw_normalized_cost: float = 0.0,
) -> MetricSet:
    """Compute all evaluation metrics from paired motion data."""
    return MetricSet(
        angle_deviation=compute_angle_deviation(paired_motion_data),
        trajectory_deviation=compute_trajectory_deviation(paired_motion_data),
        velocity_difference=compute_velocity_difference(paired_motion_data),
        smoothness_score=compute_smoothness_score(paired_motion_data),
        timing_score=compute_timing_score(paired_motion_data),
        hand_openness_deviation=compute_hand_openness_deviation(paired_motion_data),
        tool_alignment_deviation=compute_tool_alignment_deviation(paired_motion_data),
        dtw_similarity=compute_dtw_similarity(dtw_normalized_cost),
    )


def evaluate_motion_pair(
    expert_motion_source: MotionSource,
    learner_motion_source: MotionSource,
    attempt_id: str | None = None,
    db=None,
) -> EvaluationResultSchema:
    """Run the complete paired evaluation flow from motion inputs to final result."""
    expert_motion_data = load_motion_data(expert_motion_source)
    learner_motion_data = load_motion_data(learner_motion_source)
    semantic_phases = {
        "expert": analyze_video_semantics(expert_motion_data),
        "learner": analyze_video_semantics(learner_motion_data),
    }

    dtw_normalized_cost = 0.0
    if _is_modern_motion_output(expert_motion_data) and _is_modern_motion_output(learner_motion_data):
        print("Using modern aligned path: True")
        alignment_result = align_sequences(
            expert_motion=expert_motion_data,
            learner_motion=learner_motion_data,
        )
        aligned_pairs = alignment_result["aligned_pairs"]
        dtw_normalized_cost = float(alignment_result.get("dtw_normalized_cost", 0.0))
        aligned_motion_data = pair_motion_data_with_alignment(
            expert_data=expert_motion_data,
            learner_data=learner_motion_data,
            aligned_pairs=aligned_pairs,
        )
        print("DTW total cost:", alignment_result.get("dtw_total_cost"))
        print("DTW normalized cost:", dtw_normalized_cost)
        print("DTW similarity:", compute_dtw_similarity(dtw_normalized_cost))
        print("DTW path length:", alignment_result["path_length"])
        print("First 10 aligned pairs:", aligned_pairs[:10])
        print(
            "Aligned expert sequence length used for metrics:",
            _infer_metric_sequence_length(aligned_motion_data["expert_motion"]),
        )
        print(
            "Aligned learner sequence length used for metrics:",
            _infer_metric_sequence_length(aligned_motion_data["learner_motion"]),
        )
        key_error_moments = detect_key_error_moments(
            aligned_motion_data.get("modern_aligned_channels", {})
        )
    else:
        print("Using modern aligned path: False")
        logger.debug("Using legacy direct comparison path for non-modern motion inputs.")
        aligned_motion_data = pair_motion_data(
            expert_data=expert_motion_data,
            learner_data=learner_motion_data,
        )
        key_error_moments = []

    key_error_moments = map_errors_to_phases(key_error_moments, semantic_phases)

    metrics = build_metric_set(aligned_motion_data, dtw_normalized_cost=dtw_normalized_cost)
    score_result = compute_final_score(metrics.model_dump())
    final_score = score_result["score"]
    per_metric_breakdown = score_result["per_metric_breakdown"]
    print(
        "Final score input values:",
        {
            "trajectory_deviation": metrics.trajectory_deviation,
            "angle_deviation": metrics.angle_deviation,
            "velocity_difference": metrics.velocity_difference,
            "smoothness_score": metrics.smoothness_score,
            "timing_score": metrics.timing_score,
            "hand_openness_deviation": metrics.hand_openness_deviation,
            "tool_alignment_deviation": metrics.tool_alignment_deviation,
        },
    )
    print("Modern metrics used:", list(metrics.model_dump().keys()))
    print("Metric values:", metrics.model_dump())
    print("Key error moments:", key_error_moments)
    print("Semantic phases:", semantic_phases)
    print("Final score:", final_score)
    summary = build_summary(
        score=final_score,
        metrics=metrics,
        per_metric_breakdown=per_metric_breakdown,
    )
    explanation_payload = build_explanation_payload(
        {
            "score": final_score,
            "metrics": metrics,
            "per_metric_breakdown": per_metric_breakdown,
            "key_error_moments": key_error_moments,
            "semantic_phases": semantic_phases,
        }
    )
    explanation_mode = str(getattr(settings, "EXPLANATION_MODE", "rule"))
    explanation_output = generate_explanation(explanation_payload, mode=explanation_mode)
    vlm_payload = build_vlm_payload(
        score=final_score,
        metrics=metrics,
        summary=summary,
        per_metric_breakdown=per_metric_breakdown,
        key_error_moments=key_error_moments,
        semantic_phases=semantic_phases,
        explanation_payload=explanation_payload,
        explanation=explanation_output,
    )

    persisted_id = _persist_evaluation_result(
        attempt_id=attempt_id or str(uuid.uuid4()),
        score=float(final_score),
        metrics=metrics.model_dump(),
        per_metric_breakdown=_as_jsonable(per_metric_breakdown),
        key_error_moments=_as_jsonable(key_error_moments),
        semantic_phases=_as_jsonable(semantic_phases),
        explanation=explanation_output.model_dump(),
        db=db,
    )

    return EvaluationResultSchema(
        evaluation_id=persisted_id,
        score=final_score,
        metrics=metrics,
        per_metric_breakdown=per_metric_breakdown,
        key_error_moments=key_error_moments,
        semantic_phases=semantic_phases,
        explanation=explanation_output,
        summary=summary,
        vlm_payload=vlm_payload,
    )


def _is_modern_motion_output(motion_data: dict[str, Any]) -> bool:
    frames = motion_data.get("frames")
    if not isinstance(frames, list) or not frames:
        return False
    first_frame = frames[0]
    if not isinstance(first_frame, dict):
        return False
    vector = first_frame.get("flattened_feature_vector")
    return isinstance(vector, list) and len(vector) > 0


def _infer_metric_sequence_length(metric_motion: dict[str, Any]) -> int:
    for field_name in ("angle_series", "joint_trajectories", "velocity_profiles", "tool_motion"):
        mapping = metric_motion.get(field_name)
        if not isinstance(mapping, dict):
            continue
        for values in mapping.values():
            if isinstance(values, list):
                return len(values)
    return 0


async def run_evaluation_pipeline(
    expert_motion_source: MotionSource,
    learner_motion_source: MotionSource,
    attempt_id: str | None = None,
    db=None,
) -> EvaluationResultSchema:
    """Async wrapper around paired evaluation flow for API integration."""
    return evaluate_motion_pair(
        expert_motion_source=expert_motion_source,
        learner_motion_source=learner_motion_source,
        attempt_id=attempt_id,
        db=db,
    )


def _persist_evaluation_result(
    *,
    attempt_id: str,
    score: float,
    metrics: dict[str, Any],
    per_metric_breakdown: dict[str, Any],
    key_error_moments: list[dict[str, Any]],
    semantic_phases: dict[str, Any],
    explanation: dict[str, Any],
    db=None,
) -> str:
    owns_session = db is None
    session = db or SessionLocal()
    try:
        attempt_row = (
            session.query(Attempt)
            .filter(Attempt.id == str(attempt_id))
            .first()
        )
        if attempt_row is None:
            raise ValueError(f"Attempt not found for evaluation persistence: {attempt_id}")

        expert_video_row = (
            session.query(Video)
            .filter(
                Video.chapter_id == attempt_row.chapter_id,
                Video.video_role == "expert",
            )
            .order_by(Video.created_at.asc())
            .first()
        )
        if expert_video_row is None:
            raise ValueError(
                f"Expert video not found for chapter_id={attempt_row.chapter_id}"
            )

        return _insert_db_result(
            session=session,
            attempt_id=attempt_id,
            expert_video_id=str(expert_video_row.id),
            learner_video_id=str(attempt_row.learner_video_id),
            score=score,
            metrics=metrics,
            per_metric_breakdown=per_metric_breakdown,
            key_error_moments=key_error_moments,
            semantic_phases=semantic_phases,
            explanation=explanation,
        )
    finally:
        if owns_session:
            session.close()


def _as_jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _as_jsonable(val) for key, val in value.items()}
    if isinstance(value, list):
        return [_as_jsonable(item) for item in value]
    if hasattr(value, "model_dump"):
        return _as_jsonable(value.model_dump())
    return value


def _insert_db_result(
    *,
    session,
    attempt_id: str,
    expert_video_id: str,
    learner_video_id: str,
    score: float,
    metrics: dict[str, Any],
    per_metric_breakdown: dict[str, Any],
    key_error_moments: list[dict[str, Any]],
    semantic_phases: dict[str, Any],
    explanation: dict[str, Any],
) -> str:
    db_result = EvaluationModel(
        attempt_id=attempt_id,
        expert_video_id=expert_video_id,
        learner_video_id=learner_video_id,
        overall_score=score,
        status="completed",
        context_confidence=None,
        score_confidence=None,
        explanation_confidence=None,
        gate_passed=True,
        gate_reasons=[],
        metrics=metrics,
        per_metric_breakdown=per_metric_breakdown,
        key_error_moments=key_error_moments,
        semantic_phases=semantic_phases,
        summary_text=str(explanation.get("explanation", "")) if isinstance(explanation, dict) else "",
    )
    session.add(db_result)
    session.commit()
    session.refresh(db_result)
    return str(db_result.id)
