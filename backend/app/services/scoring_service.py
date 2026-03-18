"""Compute the final evaluation score from metric deviations."""

from __future__ import annotations

from typing import Any

from app.core.evaluation_constants import (
    DEFAULT_METRIC_WEIGHTS,
    EXCELLENT_SCORE_THRESHOLD,
    FAIR_SCORE_THRESHOLD,
    GOOD_SCORE_THRESHOLD,
    NEEDS_IMPROVEMENT_SCORE_THRESHOLD,
    REQUIRED_METRICS,
    SCORE_MAX,
    SCORE_MIN,
)
from app.utils.evaluation_utils import clamp, round_metric


def compute_total_deviation(metrics: dict[str, float]) -> float:
    """Compute the weighted total deviation from normalized metric values."""
    missing_metrics = [metric_name for metric_name in REQUIRED_METRICS if metric_name not in metrics]
    if missing_metrics:
        missing = ", ".join(missing_metrics)
        raise ValueError(f"Missing required metrics for scoring: {missing}")

    total_deviation = 0.0
    for metric_name, weight in DEFAULT_METRIC_WEIGHTS.items():
        metric_value = clamp(float(metrics[metric_name]), 0.0, 1.0)
        total_deviation += metric_value * weight

    return round_metric(clamp(total_deviation, 0.0, 1.0))


def classify_score(score: int) -> str:
    """Return a simple qualitative label for a numeric score."""
    if score >= EXCELLENT_SCORE_THRESHOLD:
        return "excellent"
    if score >= GOOD_SCORE_THRESHOLD:
        return "good"
    if score >= FAIR_SCORE_THRESHOLD:
        return "fair"
    if score >= NEEDS_IMPROVEMENT_SCORE_THRESHOLD:
        return "needs_improvement"
    return "poor"


def compute_final_score(metrics: dict[str, float]) -> dict[str, Any]:
    """Convert deviation metrics into a final score and label.

    Lower deviation means better performance, so the score decreases
    as the total deviation increases.
    """
    total_deviation = compute_total_deviation(metrics)
    raw_score = SCORE_MAX * (1.0 - total_deviation)
    final_score = int(round(clamp(raw_score, SCORE_MIN, SCORE_MAX)))

    return {
        "score": final_score,
        "label": classify_score(final_score),
    }


def compute_score(metrics: dict[str, float]) -> int:
    """Backward-compatible helper that returns only the numeric score."""
    return int(compute_final_score(metrics)["score"])
