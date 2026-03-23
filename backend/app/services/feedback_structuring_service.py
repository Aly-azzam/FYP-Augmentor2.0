"""Build simple structured feedback from evaluation results."""

from __future__ import annotations

from app.schemas.evaluation_schema import EvaluationSummary, MetricSet, VLMInputPayload

METRIC_LABELS = {
    "angle_deviation": "joint angle control",
    "trajectory_deviation": "trajectory matching",
    "velocity_difference": "movement speed consistency",
    "tool_alignment_deviation": "tool alignment",
}

FOCUS_LABELS = {
    "angle_deviation": "Improve joint angle control",
    "trajectory_deviation": "Improve movement trajectory control",
    "velocity_difference": "Improve speed consistency",
    "tool_alignment_deviation": "Improve tool alignment control",
}


def _get_strength_prefix(score: int) -> str:
    """Return a simple wording prefix based on the final score."""
    if score >= 75:
        return "Good"
    return "Best area:"


def get_best_metric(metrics: MetricSet) -> str:
    """Return the metric name with the lowest deviation."""
    metric_values = metrics.model_dump()
    return min(metric_values, key=metric_values.get)


def get_worst_metric(metrics: MetricSet) -> str:
    """Return the metric name with the highest deviation."""
    metric_values = metrics.model_dump()
    return max(metric_values, key=metric_values.get)


def build_summary(score: int, metrics: MetricSet) -> EvaluationSummary:
    """Build a simple structured summary from score and metric values."""
    best_metric = get_best_metric(metrics)
    worst_metric = get_worst_metric(metrics)

    main_strength = f"{_get_strength_prefix(score)} {METRIC_LABELS[best_metric]}"
    main_weakness = f"{METRIC_LABELS[worst_metric].capitalize()} needs improvement"
    focus_area = FOCUS_LABELS[worst_metric]

    return EvaluationSummary(
        main_strength=main_strength,
        main_weakness=main_weakness,
        focus_area=focus_area,
    )


def build_vlm_payload(score: int, metrics: MetricSet, summary: EvaluationSummary) -> VLMInputPayload:
    """Build the payload to be passed later to the AI explanation layer."""
    return VLMInputPayload(
        score=score,
        metrics=metrics,
        summary=summary,
    )


def structure_feedback(score: int, metrics: MetricSet) -> dict:
    """Convenience helper returning both summary and VLM payload."""
    summary = build_summary(score=score, metrics=metrics)
    vlm_payload = build_vlm_payload(score=score, metrics=metrics, summary=summary)
    return {
        "summary": summary,
        "vlm_payload": vlm_payload,
    }
