"""Feedback Structuring Service — build structured evaluation summary.

Responsibilities:
- generate strengths/weaknesses summary from metrics
- identify areas for improvement
- prepare structured data for VLM and frontend

Owner: Evaluation Engine (Person 3)
"""


async def structure_feedback(score: int, metrics: dict, warnings: list) -> dict:
    """Build structured feedback from evaluation results.

    Returns dict with summary text, strengths, weaknesses.
    """
    # TODO: implement rule-based feedback structuring
    return {
        "summary": f"Overall score: {score}/100",
        "strengths": [],
        "weaknesses": [],
    }
