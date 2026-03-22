import asyncio

from app.core.evaluation_constants import SCORE_MAX, SCORE_MIN
from app.services.scoring_service import compute_score


def test_compute_score_returns_placeholder_value_within_bounds() -> None:
    metrics = {
        "angle_deviation": 10.0,
        "trajectory_deviation": 12.0,
        "velocity_difference": 8.0,
        "tool_alignment_deviation": 6.0,
    }

    result = asyncio.run(compute_score(metrics))

    assert isinstance(result, int)
    assert SCORE_MIN <= result <= SCORE_MAX
    assert result == 75
