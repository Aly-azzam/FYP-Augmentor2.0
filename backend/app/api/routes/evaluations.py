from uuid import UUID

from fastapi import APIRouter

from app.schemas.upload_schema import (
    EvaluationStartRequest,
    EvaluationStartResponse,
    EvaluationStatusResponse,
)
from app.schemas.evaluation_schema import EvaluationResultOut, EvaluationMetrics

router = APIRouter(prefix="/api/evaluations", tags=["Evaluations"])


@router.post("/start", response_model=EvaluationStartResponse)
async def start_evaluation(request: EvaluationStartRequest):
    """Start async evaluation pipeline for a learner attempt.

    Sets attempt status to queued and dispatches processing job.
    """
    # TODO: implement real flow:
    # 1. Verify attempt exists and is in uploaded status
    # 2. Set status to queued
    # 3. Dispatch async pipeline job
    # 4. Create EvaluationResult record

    evaluation_id = UUID("00000000-0000-0000-0000-000000000002")

    return EvaluationStartResponse(
        evaluation_id=evaluation_id,
        attempt_id=request.attempt_id,
        status="queued",
        message="Evaluation queued for processing.",
    )


@router.get("/{evaluation_id}/status", response_model=EvaluationStatusResponse)
async def get_evaluation_status(evaluation_id: UUID):
    """Poll evaluation processing status."""
    # TODO: query real status from DB
    return EvaluationStatusResponse(
        evaluation_id=evaluation_id,
        attempt_id=UUID("00000000-0000-0000-0000-000000000000"),
        status="running",
        current_stage="perception",
        message="Processing in progress...",
    )


@router.get("/{evaluation_id}/result", response_model=EvaluationResultOut)
async def get_evaluation_result(evaluation_id: UUID):
    """Get final evaluation result. Only available when status is terminal."""
    # TODO: query real result from DB, return 404/409 if not ready
    return EvaluationResultOut(
        evaluation_id=evaluation_id,
        chapter_id=UUID("00000000-0000-0000-0000-000000000000"),
        expert_video_id=UUID("00000000-0000-0000-0000-000000000001"),
        learner_video_id=UUID("00000000-0000-0000-0000-000000000000"),
        status="completed",
        score=75,
        metrics=EvaluationMetrics(
            angle_deviation=8.5,
            trajectory_deviation=12.3,
            velocity_difference=15.0,
            tool_alignment_deviation=6.2,
        ),
        summary="Placeholder evaluation summary",
        ai_text="Placeholder AI explanation",
    )
