from __future__ import annotations

import asyncio
from pathlib import Path

from app.schemas.evaluation_schema import EvaluationResult
from app.services.evaluation_engine_service import run_evaluation_pipeline


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
    assert hasattr(result.metrics, "tool_alignment_deviation")


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
