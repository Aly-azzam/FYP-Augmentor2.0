"""Constants used by the Evaluation Engine.

This file keeps the metric weights and score thresholds in one place so
they are easy to read now and easy to tune later.
"""

DEVIATION_METRICS = [
    "trajectory_deviation",
    "angle_deviation",
    "velocity_difference",
    "hand_openness_deviation",
    "tool_alignment_deviation",
]

QUALITY_METRICS = [
    "smoothness_score",
    "timing_score",
    "dtw_similarity",
]

ACTIVE_SCORING_METRICS = [
    "trajectory_deviation",
    "angle_deviation",
    "velocity_difference",
    "smoothness_score",
    "timing_score",
    "hand_openness_deviation",
    "dtw_similarity",
]

REQUIRED_METRICS = ACTIVE_SCORING_METRICS[:]

# Metric weights for the final score calculation.
# These should add up to 1.0.
TRAJECTORY_DEVIATION_WEIGHT = 0.20
ANGLE_DEVIATION_WEIGHT = 0.15
VELOCITY_DIFFERENCE_WEIGHT = 0.15
SMOOTHNESS_SCORE_WEIGHT = 0.05
TIMING_SCORE_WEIGHT = 0.05
HAND_OPENNESS_DEVIATION_WEIGHT = 0.10
DTW_SIMILARITY_WEIGHT = 0.30

DEFAULT_METRIC_WEIGHTS = {
    "trajectory_deviation": TRAJECTORY_DEVIATION_WEIGHT,
    "angle_deviation": ANGLE_DEVIATION_WEIGHT,
    "velocity_difference": VELOCITY_DIFFERENCE_WEIGHT,
    "smoothness_score": SMOOTHNESS_SCORE_WEIGHT,
    "timing_score": TIMING_SCORE_WEIGHT,
    "hand_openness_deviation": HAND_OPENNESS_DEVIATION_WEIGHT,
    "dtw_similarity": DTW_SIMILARITY_WEIGHT,
}

DTW_SIMILARITY_DECAY = 0.5

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
