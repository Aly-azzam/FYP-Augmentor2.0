"""Build simple structured feedback from evaluation results."""

from __future__ import annotations

from app.schemas.evaluation_schema import (
    ExplanationOutput,
    EvaluationSummary,
    KeyErrorMoment,
    MetricBreakdownItem,
    MetricSet,
    SemanticPhasesOut,
    VLMInputPayload,
)

METRIC_LABELS = {
    "angle_deviation": "joint angle control",
    "trajectory_deviation": "movement trajectory",
    "velocity_difference": "speed consistency",
    "smoothness_score": "motion smoothness",
    "timing_score": "timing and rhythm",
    "hand_openness_deviation": "hand openness control",
    "tool_alignment_deviation": "tool alignment",
    "dtw_similarity": "overall motion similarity",
}

FOCUS_LABELS = {
    "angle_deviation": "Improve joint angle control",
    "trajectory_deviation": "Improve movement trajectory control",
    "velocity_difference": "Improve speed consistency",
    "smoothness_score": "Improve motion smoothness",
    "timing_score": "Improve timing and rhythm",
    "hand_openness_deviation": "Improve hand openness control",
    "tool_alignment_deviation": "Improve tool alignment control",
    "dtw_similarity": "Improve overall motion pattern",
}
STRENGTH_PHRASES = {
    "angle_deviation": "Your joint angle control is accurate.",
    "trajectory_deviation": "Your movement trajectory is close to the expert.",
    "velocity_difference": "Your speed consistency is close to the expert.",
    "smoothness_score": "Your motion is smooth and stable.",
    "timing_score": "Your timing is consistent with the expert.",
    "hand_openness_deviation": "Your hand openness control is strong.",
    "tool_alignment_deviation": "Your tool alignment is stable.",
    "dtw_similarity": "Your overall motion pattern closely matches the expert.",
}
ADVICE_BY_METRIC = {
    "trajectory_deviation": "Focus on following the expert's movement path more closely.",
    "timing_score": "Try to match the timing of the expert more precisely.",
    "angle_deviation": "Pay attention to your wrist and finger positioning.",
    "velocity_difference": "Practice keeping your movement speed more consistent.",
    "smoothness_score": "Work on smoother transitions between movement phases.",
    "hand_openness_deviation": "Focus on maintaining consistent hand openness and grip shape.",
    "tool_alignment_deviation": "Focus on keeping your tool alignment stable throughout the motion.",
    "dtw_similarity": "Try to match the expert's overall motion pattern more closely.",
}

HIGHER_IS_BETTER = {"smoothness_score", "timing_score", "dtw_similarity"}


def _get_strength_prefix(score: int) -> str:
    """Return a simple wording prefix based on the final score."""
    if score >= 90:
        return "Excellent"
    if score >= 75:
        return "Good"
    return "Developing"


def _metric_quality(metric_name: str, metric_value: float) -> float:
    value = float(metric_value)
    if metric_name in HIGHER_IS_BETTER:
        return max(0.0, min(1.0, value))
    return max(0.0, min(1.0, 1.0 - value))


def get_best_metric(
    metrics: MetricSet,
    per_metric_breakdown: dict[str, MetricBreakdownItem] | None = None,
) -> str:
    """Return best metric by normalized quality."""
    if per_metric_breakdown:
        return max(per_metric_breakdown.keys(), key=lambda name: _breakdown_quality(per_metric_breakdown[name]))
    metric_values = metrics.model_dump()
    return max(metric_values.keys(), key=lambda name: _metric_quality(name, metric_values[name]))


def get_worst_metric(
    metrics: MetricSet,
    per_metric_breakdown: dict[str, MetricBreakdownItem] | None = None,
) -> str:
    """Return weakest metric by normalized quality."""
    if per_metric_breakdown:
        return min(per_metric_breakdown.keys(), key=lambda name: _breakdown_quality(per_metric_breakdown[name]))
    metric_values = metrics.model_dump()
    return min(metric_values.keys(), key=lambda name: _metric_quality(name, metric_values[name]))


def build_summary(
    score: int,
    metrics: MetricSet,
    per_metric_breakdown: dict[str, MetricBreakdownItem] | None = None,
) -> EvaluationSummary:
    """Build a simple structured summary from score and metric values."""
    best_metric = get_best_metric(metrics, per_metric_breakdown=per_metric_breakdown)
    worst_metric = get_worst_metric(metrics, per_metric_breakdown=per_metric_breakdown)

    main_strength = f"{_get_strength_prefix(score)} {METRIC_LABELS.get(best_metric, best_metric)}"
    main_weakness = f"{METRIC_LABELS.get(worst_metric, worst_metric).capitalize()} needs improvement"
    focus_area = FOCUS_LABELS.get(worst_metric, f"Improve {worst_metric}")

    return EvaluationSummary(
        main_strength=main_strength,
        main_weakness=main_weakness,
        focus_area=focus_area,
    )


def _breakdown_quality(item: MetricBreakdownItem | dict) -> float:
    if isinstance(item, dict):
        return float(item.get("normalized_quality", 0.0))
    return float(item.normalized_quality)


def build_vlm_payload(
    score: int,
    metrics: MetricSet,
    summary: EvaluationSummary,
    per_metric_breakdown: dict[str, MetricBreakdownItem] | None = None,
    key_error_moments: list[KeyErrorMoment] | None = None,
    semantic_phases: dict[str, SemanticPhasesOut] | None = None,
    explanation_payload: dict | None = None,
    explanation: ExplanationOutput | None = None,
) -> VLMInputPayload:
    """Build the payload to be passed later to the AI explanation layer."""
    return VLMInputPayload(
        score=score,
        metrics=metrics,
        summary=summary,
        per_metric_breakdown=per_metric_breakdown or {},
        key_error_moments=key_error_moments or [],
        semantic_phases=semantic_phases or {},
        explanation_payload=explanation_payload or {},
        explanation=explanation
        or ExplanationOutput(
            mode="rule_based",
            explanation="",
            strengths=[],
            weaknesses=[],
            advice="",
        ),
    )


def structure_feedback(
    score: int,
    metrics: MetricSet,
    per_metric_breakdown: dict[str, MetricBreakdownItem] | None = None,
    key_error_moments: list[KeyErrorMoment] | None = None,
    semantic_phases: dict[str, SemanticPhasesOut] | None = None,
    explanation_payload: dict | None = None,
    explanation: ExplanationOutput | None = None,
) -> dict:
    """Convenience helper returning both summary and VLM payload."""
    summary = build_summary(score=score, metrics=metrics, per_metric_breakdown=per_metric_breakdown)
    vlm_payload = build_vlm_payload(
        score=score,
        metrics=metrics,
        summary=summary,
        per_metric_breakdown=per_metric_breakdown,
        key_error_moments=key_error_moments,
        semantic_phases=semantic_phases,
        explanation_payload=explanation_payload,
        explanation=explanation,
    )
    return {
        "summary": summary,
        "vlm_payload": vlm_payload,
    }


def build_explanation_payload(evaluation_result) -> dict:
    """Build model-agnostic explanation payload from factual evaluation outputs."""
    payload = _as_mapping(evaluation_result)
    breakdown = payload.get("per_metric_breakdown", {})
    best_metric = _best_metric_from_breakdown(breakdown)
    worst_metric = _worst_metric_from_breakdown(breakdown)
    return {
        "score": int(payload.get("score", 0)),
        "metrics": payload.get("metrics", {}),
        "per_metric_breakdown": breakdown,
        "key_error_moments": payload.get("key_error_moments", []),
        "semantic_phases": payload.get("semantic_phases", {}),
        "strength": METRIC_LABELS.get(best_metric, best_metric),
        "weakness": METRIC_LABELS.get(worst_metric, worst_metric),
    }


def generate_rule_based_explanation(payload: dict) -> ExplanationOutput:
    """Create deterministic grounded rule-based explanation from payload."""
    score = int(payload.get("score", 0))
    breakdown = payload.get("per_metric_breakdown", {})
    error_moments = payload.get("key_error_moments", [])

    if score >= 85:
        overall = "Your performance is very close to the expert."
    elif score >= 70:
        overall = "Your performance is good but still has noticeable differences."
    else:
        overall = "Your performance differs significantly from the expert."

    best_metric = _best_metric_from_breakdown(breakdown)
    worst_metric = _worst_metric_from_breakdown(breakdown)
    strengths = [STRENGTH_PHRASES.get(best_metric, "You show a clear area of strength.")]
    advice = ADVICE_BY_METRIC.get(worst_metric, "Focus on improving your weakest metric areas.")

    ranked_errors = _rank_error_moments(error_moments)
    weaknesses: list[str] = []
    for error in ranked_errors[:2]:
        semantic_label = str(error.get("semantic_label", "")).strip()
        if not semantic_label:
            semantic_label = str(error.get("label", "error pattern")).strip()
        weaknesses.append(f"Your movement shows {semantic_label}.")
    if not weaknesses:
        weaknesses.append("No major localized error segment was detected.")

    explanation_text = " ".join([overall] + strengths + weaknesses + [advice]).strip()
    return ExplanationOutput(
        mode="rule_based",
        explanation=explanation_text,
        strengths=strengths,
        weaknesses=weaknesses,
        advice=advice,
    )


def generate_vlm_explanation(payload: dict) -> ExplanationOutput:
    """Placeholder for future VLM-based explanation generation."""
    _ = payload
    return ExplanationOutput(
        mode="vlm_placeholder",
        explanation="VLM explanation not implemented yet",
        strengths=[],
        weaknesses=[],
        advice="",
    )


def generate_explanation(evaluation_result, mode: str = "rule") -> ExplanationOutput:
    """Dispatch explanation generation by mode using a shared payload."""
    payload = build_explanation_payload(evaluation_result)
    if mode == "vlm":
        return generate_vlm_explanation(payload)
    return generate_rule_based_explanation(payload)


def _as_mapping(value) -> dict:
    if isinstance(value, dict):
        return value
    if hasattr(value, "model_dump"):
        dumped = value.model_dump()
        if isinstance(dumped, dict):
            return dumped
    return {}


def _best_metric_from_breakdown(breakdown) -> str:
    if isinstance(breakdown, dict) and breakdown:
        return max(
            breakdown.keys(),
            key=lambda name: float(_as_mapping(breakdown[name]).get("normalized_quality", 0.0)),
        )
    return "trajectory_deviation"


def _worst_metric_from_breakdown(breakdown) -> str:
    if isinstance(breakdown, dict) and breakdown:
        return min(
            breakdown.keys(),
            key=lambda name: float(_as_mapping(breakdown[name]).get("normalized_quality", 0.0)),
        )
    return "trajectory_deviation"


def _rank_error_moments(error_moments) -> list[dict]:
    normalized = [_as_mapping(item) for item in error_moments] if isinstance(error_moments, list) else []
    return sorted(
        normalized,
        key=lambda item: (-float(item.get("severity", 0.0)), str(item.get("error_type", ""))),
    )
