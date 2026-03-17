"""Scoring Service — compute composite score from metrics.

Owner: Evaluation Engine (Person 3)
"""

from app.core.evaluation_constants import DEFAULT_METRIC_WEIGHTS, SCORE_MIN, SCORE_MAX


async def compute_score(metrics: dict[str, float]) -> int:
    """Compute weighted composite score from individual metrics.

    Lower metric values = better performance = higher score.
    Returns score clamped to [SCORE_MIN, SCORE_MAX].
    """
    # TODO: implement real scoring formula based on metric ranges
    # Stub: return a placeholder score
    return 75
