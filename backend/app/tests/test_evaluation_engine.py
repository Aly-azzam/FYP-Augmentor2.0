from __future__ import annotations

import asyncio
from pathlib import Path

from app.schemas.evaluation_schema import EvaluationResult
from app.services.comparison_service import map_errors_to_phases
from app.services.evaluation_engine_service import run_evaluation_pipeline
from app.services.feedback_structuring_service import (
    build_explanation_payload,
    generate_explanation,
    generate_rule_based_explanation,
)


BACKEND_DIR = Path(__file__).resolve().parents[2]
MOTION_DIR = BACKEND_DIR / "storage" / "processed" / "motion"
EXPERT_MOCK_PATH = MOTION_DIR / "expert_mock.json"
LEARNER_MOCK_PATH = MOTION_DIR / "learner_mock.json"


def run_pipeline() -> EvaluationResult:
    return asyncio.run(
        run_evaluation_pipeline(
            expert_motion_source=EXPERT_MOCK_PATH,
            learner_motion_source=LEARNER_MOCK_PATH,
        )
    )


def _build_modern_motion_payload(video_id: str, scalar_series: list[float]) -> dict:
    frames = []
    for idx, scalar in enumerate(scalar_series):
        hand_features = {
            "joint_angles": {
                "wrist_index_mcp_index_tip": float(40.0 + scalar),
                "wrist_middle_mcp_middle_tip": float(50.0 + scalar),
                "wrist_ring_mcp_ring_tip": float(55.0 + scalar),
                "wrist_index_mcp_pinky_mcp": float(60.0 + scalar),
            },
            "wrist_position": [float(scalar), float(scalar * 0.5), 0.0],
            "palm_center": [float(scalar * 0.8), float(scalar * 0.4), 0.0],
            "fingertip_positions": {"index_tip": [float(scalar * 1.2), float(scalar * 0.6), 0.0]},
            "wrist_velocity": [float(scalar * 0.2), float(scalar * 0.1), 0.0],
            "palm_velocity": [float(scalar * 0.15), float(scalar * 0.08), 0.0],
            "hand_openness": float(0.5 + scalar * 0.01),
            "pinch_distance": float(0.2 + scalar * 0.01),
            "finger_spread": float(0.3 + scalar * 0.01),
        }
        frames.append(
            {
                "frame_index": idx,
                "timestamp_sec": idx / 30.0,
                "flattened_feature_vector": [float(scalar)] * 108,
                "left_hand_features": hand_features,
                "right_hand_features": hand_features,
            }
        )
    return {
        "video_id": video_id,
        "fps": 30.0,
        "total_frames": len(frames),
        "sequence_length": len(frames),
        "frames": frames,
    }


def test_evaluation_pipeline_returns_evaluation_result() -> None:
    result = run_pipeline()
    assert isinstance(result, EvaluationResult)


def test_score_is_between_zero_and_one_hundred() -> None:
    result = run_pipeline()
    assert 0 <= result.score <= 100


def test_all_metric_fields_exist() -> None:
    result = run_pipeline()

    assert hasattr(result.metrics, "angle_deviation")
    assert hasattr(result.metrics, "trajectory_deviation")
    assert hasattr(result.metrics, "velocity_difference")
    assert hasattr(result.metrics, "smoothness_score")
    assert hasattr(result.metrics, "timing_score")
    assert hasattr(result.metrics, "hand_openness_deviation")
    assert hasattr(result.metrics, "tool_alignment_deviation")
    assert isinstance(result.per_metric_breakdown, dict)
    assert "trajectory_deviation" in result.per_metric_breakdown
    assert "angle_deviation" in result.per_metric_breakdown
    assert "velocity_difference" in result.per_metric_breakdown
    assert "smoothness_score" in result.per_metric_breakdown
    assert "timing_score" in result.per_metric_breakdown


def test_summary_fields_exist() -> None:
    result = run_pipeline()

    assert hasattr(result.summary, "main_strength")
    assert hasattr(result.summary, "main_weakness")
    assert hasattr(result.summary, "focus_area")


def test_vlm_payload_exists() -> None:
    result = run_pipeline()

    assert result.vlm_payload is not None
    assert result.vlm_payload.score == result.score
    assert result.vlm_payload.metrics == result.metrics
    assert result.vlm_payload.summary == result.summary
    assert result.vlm_payload.per_metric_breakdown == result.per_metric_breakdown
    assert result.vlm_payload.semantic_phases == result.semantic_phases
    assert result.vlm_payload.explanation == result.explanation
    assert "score" in result.vlm_payload.explanation_payload


def test_modern_path_includes_key_error_moments() -> None:
    expert_motion = _build_modern_motion_payload("expert", [0.0, 1.0, 2.0, 3.0])
    learner_motion = _build_modern_motion_payload("learner", [0.0, 0.5, 1.0, 1.5, 2.5, 3.0])
    result = asyncio.run(
        run_evaluation_pipeline(
            expert_motion_source=expert_motion,
            learner_motion_source=learner_motion,
        )
    )

    assert len(result.key_error_moments) == 3
    assert result.vlm_payload.key_error_moments == result.key_error_moments

    types = {moment.error_type for moment in result.key_error_moments}
    assert types == {"angle_error", "trajectory_error", "speed_error"}

    for moment in result.key_error_moments:
        assert moment.start_aligned_index <= moment.end_aligned_index
        assert moment.start_expert_frame <= moment.end_expert_frame
        assert moment.start_learner_frame <= moment.end_learner_frame
        assert 0.0 <= moment.severity <= 1.0
        assert moment.semantic_phase is not None
        assert moment.semantic_label is not None
        assert moment.semantic_phase.label in moment.semantic_label

    result_again = asyncio.run(
        run_evaluation_pipeline(
            expert_motion_source=expert_motion,
            learner_motion_source=learner_motion,
        )
    )
    assert result_again.key_error_moments == result.key_error_moments


def test_semantic_phases_exist_and_are_valid() -> None:
    result = run_pipeline()
    assert "expert" in result.semantic_phases
    assert "learner" in result.semantic_phases

    expert_semantics = result.semantic_phases["expert"]
    learner_semantics = result.semantic_phases["learner"]
    assert len(expert_semantics.phases) >= 1
    assert len(learner_semantics.phases) >= 1

    for phase in expert_semantics.phases + learner_semantics.phases:
        assert isinstance(phase.label, str)
        assert phase.label.strip() != ""
        assert phase.start_frame <= phase.end_frame


def test_error_phase_mapping_no_crash_when_phases_missing() -> None:
    error_moments = [
        {
            "error_type": "trajectory_error",
            "label": "path drift",
            "start_aligned_index": 1,
            "end_aligned_index": 2,
            "start_expert_frame": 10,
            "end_expert_frame": 20,
            "start_learner_frame": 11,
            "end_learner_frame": 21,
            "severity": 0.4,
        }
    ]
    fused = map_errors_to_phases(error_moments, {})
    assert len(fused) == 1
    assert fused[0]["semantic_phase"] is not None
    assert isinstance(fused[0]["semantic_label"], str)


def test_grounded_explanation_exists_and_uses_semantic_labels() -> None:
    result = run_pipeline()
    assert isinstance(result.explanation.explanation, str)
    assert result.explanation.explanation.strip() != ""
    assert isinstance(result.explanation.strengths, list)
    assert isinstance(result.explanation.weaknesses, list)

    if result.key_error_moments:
        semantic_label = result.key_error_moments[0].semantic_label
        if semantic_label:
            joined = " ".join(result.explanation.weaknesses)
            assert semantic_label in joined


def test_explanation_is_deterministic_for_same_input() -> None:
    result_a = run_pipeline()
    result_b = run_pipeline()
    assert result_a.explanation == result_b.explanation


def test_explanation_payload_builder_contains_expected_fields() -> None:
    result = run_pipeline()
    payload = build_explanation_payload(result)
    for field in (
        "score",
        "metrics",
        "per_metric_breakdown",
        "key_error_moments",
        "semantic_phases",
        "strength",
        "weakness",
    ):
        assert field in payload


def test_rule_based_explanation_from_payload_works() -> None:
    payload = build_explanation_payload(run_pipeline())
    explanation = generate_rule_based_explanation(payload)
    assert explanation.mode == "rule_based"
    assert isinstance(explanation.explanation, str)
    assert explanation.explanation.strip() != ""


def test_generate_explanation_dispatch_modes() -> None:
    result = run_pipeline()
    rule_output = generate_explanation(result, mode="rule")
    vlm_output = generate_explanation(result, mode="vlm")

    assert rule_output.mode == "rule_based"
    assert rule_output.explanation.strip() != ""
    assert vlm_output.mode == "vlm_placeholder"
    assert vlm_output.explanation == "VLM explanation not implemented yet"
