"""Constants used by the Evaluation Engine.

This file keeps the metric weights and score thresholds in one place so
they are easy to read now and easy to tune later.
"""

REQUIRED_METRICS = [
    "angle_deviation",
    "trajectory_deviation",
    "velocity_difference",
    "tool_alignment_deviation",
]

# Metric weights for the final score calculation.
# These should add up to 1.0.
ANGLE_DEVIATION_WEIGHT = 0.30
TRAJECTORY_DEVIATION_WEIGHT = 0.30
VELOCITY_DIFFERENCE_WEIGHT = 0.20
TOOL_ALIGNMENT_DEVIATION_WEIGHT = 0.20

DEFAULT_METRIC_WEIGHTS = {
    "angle_deviation": ANGLE_DEVIATION_WEIGHT,
    "trajectory_deviation": TRAJECTORY_DEVIATION_WEIGHT,
    "velocity_difference": VELOCITY_DIFFERENCE_WEIGHT,
    "tool_alignment_deviation": TOOL_ALIGNMENT_DEVIATION_WEIGHT,
}

SCORE_MIN = 0
SCORE_MAX = 100

# Qualitative score thresholds for summaries or UI labels.
EXCELLENT_SCORE_THRESHOLD = 90
GOOD_SCORE_THRESHOLD = 75
FAIR_SCORE_THRESHOLD = 60
NEEDS_IMPROVEMENT_SCORE_THRESHOLD = 40

QUALITATIVE_SCORE_LABELS = {
    "excellent": EXCELLENT_SCORE_THRESHOLD,
    "good": GOOD_SCORE_THRESHOLD,
    "fair": FAIR_SCORE_THRESHOLD,
    "needs_improvement": NEEDS_IMPROVEMENT_SCORE_THRESHOLD,
}
