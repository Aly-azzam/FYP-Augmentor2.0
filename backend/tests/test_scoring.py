from app.core.evaluation_constants import SCORE_MAX, SCORE_MIN
from app.services.scoring_service import compute_final_score


def _sample_metrics() -> dict[str, float]:
    return {
        "trajectory_deviation": 0.1694,
        "angle_deviation": 0.0006,
        "velocity_difference": 0.0422,
        "smoothness_score": 0.8032,
        "timing_score": 0.8,
        "hand_openness_deviation": 0.0,
        "tool_alignment_deviation": 0.0,
    }


def test_final_score_is_within_bounds() -> None:
    result = compute_final_score(_sample_metrics())
    assert SCORE_MIN <= result["score"] <= SCORE_MAX


def test_scoring_is_deterministic_for_same_inputs() -> None:
    metrics = _sample_metrics()
    first = compute_final_score(metrics)
    second = compute_final_score(metrics)
    assert first["score"] == second["score"]
    assert first["per_metric_breakdown"] == second["per_metric_breakdown"]


def test_per_metric_breakdown_exists_and_contributions_non_negative() -> None:
    result = compute_final_score(_sample_metrics())
    breakdown = result["per_metric_breakdown"]
    assert isinstance(breakdown, dict)
    assert "trajectory_deviation" in breakdown
    assert "smoothness_score" in breakdown
    for item in breakdown.values():
        assert item["contribution"] >= 0.0


def test_contribution_sum_matches_score_approximately() -> None:
    result = compute_final_score(_sample_metrics())
    contribution_sum = result["contribution_sum"]
    assert abs(contribution_sum - result["score"]) <= 1.0


def test_deviation_metrics_are_inverted_and_quality_direct() -> None:
    result = compute_final_score(_sample_metrics())
    breakdown = result["per_metric_breakdown"]

    assert breakdown["trajectory_deviation"]["normalized_quality"] == round(1.0 - 0.1694, 4)
    assert breakdown["angle_deviation"]["normalized_quality"] == round(1.0 - 0.0006, 4)
    assert breakdown["velocity_difference"]["normalized_quality"] == round(1.0 - 0.0422, 4)
    assert breakdown["smoothness_score"]["normalized_quality"] == round(0.8032, 4)
    assert breakdown["timing_score"]["normalized_quality"] == round(0.8, 4)
