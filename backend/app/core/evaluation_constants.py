"""Constants for the Evaluation Engine."""

REQUIRED_METRICS = [
    "angle_deviation",
    "trajectory_deviation",
    "velocity_difference",
    "tool_alignment_deviation",
]

SCORE_MIN = 0
SCORE_MAX = 100

# Weights for composite score (must sum to 1.0)
DEFAULT_METRIC_WEIGHTS = {
    "angle_deviation": 0.25,
    "trajectory_deviation": 0.30,
    "velocity_difference": 0.20,
    "tool_alignment_deviation": 0.25,
}
