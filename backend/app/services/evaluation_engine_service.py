"""Evaluation Engine Service — orchestrates the full evaluation pipeline.

Fixed pipeline order:
Upload → Perception → Motion Representation → Evaluation → VLM → Final Result

This service coordinates the async execution of all stages
and manages the LearnerAttempt status transitions.
"""

import uuid
from typing import Optional

from app.core.constants import AttemptStatus


async def run_evaluation_pipeline(
    attempt_id: uuid.UUID,
    learner_video_path: str,
    expert_video_path: str,
    chapter_id: uuid.UUID,
) -> dict:
    """Run the full evaluation pipeline for a learner attempt.

    Steps:
    1. Set status to RUNNING
    2. Run Perception on both expert and learner videos
    3. Run Motion Representation
    4. Run Comparison + Metrics
    5. Compute Score
    6. Structure Feedback
    7. Generate VLM Explanation
    8. Persist EvaluationResult
    9. Set final status (completed / completed_with_warnings / failed)

    Returns dict with evaluation result data.
    """
    # TODO: wire all real sub-services and implement status transitions
    return {
        "status": AttemptStatus.COMPLETED,
        "score": 75,
        "metrics": {
            "angle_deviation": 8.5,
            "trajectory_deviation": 12.3,
            "velocity_difference": 15.0,
            "tool_alignment_deviation": 6.2,
        },
        "summary": "Placeholder evaluation result",
        "ai_text": "Placeholder AI explanation",
        "warnings": [],
    }
