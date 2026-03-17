"""Trajectory Metrics Service — compute trajectory deviation.

Canonical unit: normalized coordinate distance (degrees if angular)

Owner: Evaluation Engine (Person 3)
"""


async def compute_trajectory_deviation(expert_data: dict, learner_data: dict) -> float:
    """Compute trajectory deviation between expert and learner paths.

    Returns deviation value.
    """
    # TODO: implement trajectory comparison (e.g. DTW distance, Frechet)
    return 0.0
