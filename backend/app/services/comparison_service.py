"""Comparison Service — compare expert vs learner motion data.

Responsibilities:
- receive aligned motion pairs
- dispatch to individual metric services
- aggregate metric results

Owner: Evaluation Engine (Person 3)
"""


async def compare_motions(alignment_data: dict) -> dict:
    """Compare expert and learner aligned motion data.

    Returns dict with per-metric raw values.
    """
    # TODO: dispatch to angle, trajectory, velocity, tool metric services
    return {
        "angle_deviation": 0.0,
        "trajectory_deviation": 0.0,
        "velocity_difference": 0.0,
        "tool_alignment_deviation": 0.0,
    }
