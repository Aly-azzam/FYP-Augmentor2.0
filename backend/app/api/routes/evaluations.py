from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.models.evaluation_result import EvaluationResult as EvaluationResultModel
from app.schemas.upload_schema import (
    EvaluationStartRequest,
    EvaluationStartResponse,
    EvaluationStatusResponse,
)
from app.schemas.evaluation_schema import (
    AttemptProgressOut,
    EvaluationHistoryOut,
    EvaluationMetrics,
    EvaluationResultOut,
    PersistedEvaluationOut,
)
from app.services.progress_service import compute_progress

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
            smoothness_score=0.7,
            timing_score=0.65,
            hand_openness_deviation=0.2,
            tool_alignment_deviation=6.2,
        ),
        summary="Placeholder evaluation summary",
        ai_text="Placeholder AI explanation",
    )


@router.get("/history/{attempt_id}", response_model=EvaluationHistoryOut)
async def get_evaluation_history(attempt_id: str, db: Session = Depends(get_db)):
    rows = (
        db.query(EvaluationResultModel)
        .filter(EvaluationResultModel.attempt_id == attempt_id)
        .order_by(EvaluationResultModel.created_at.desc())
        .all()
    )
    return EvaluationHistoryOut(evaluations=[_to_persisted_out(row) for row in rows])


@router.get("/progress/{attempt_id}", response_model=AttemptProgressOut)
async def get_evaluation_progress(attempt_id: str, db: Session = Depends(get_db)):
    rows = (
        db.query(EvaluationResultModel)
        .filter(EvaluationResultModel.attempt_id == attempt_id)
        .order_by(EvaluationResultModel.created_at.desc())
        .all()
    )
    evaluations = [_to_persisted_out(row).model_dump() for row in rows]
    computed = compute_progress(evaluations)
    return AttemptProgressOut(attempt_id=attempt_id, **computed)


@router.get("/{evaluation_id}", response_model=PersistedEvaluationOut)
async def get_evaluation_by_id(evaluation_id: str, db: Session = Depends(get_db)):
    row = db.query(EvaluationResultModel).filter(EvaluationResultModel.id == evaluation_id).first()
    if row is None:
        raise HTTPException(status_code=404, detail="Evaluation not found")
    return _to_persisted_out(row)


def _to_persisted_out(row: EvaluationResultModel) -> PersistedEvaluationOut:
    return PersistedEvaluationOut(
        id=str(row.id),
        attempt_id=str(row.attempt_id),
        score=float(row.score or 0.0),
        metrics=row.metrics or {},
        per_metric_breakdown=row.per_metric_breakdown or {},
        key_error_moments=row.key_error_moments or [],
        semantic_phases=row.semantic_phases or {},
        explanation=row.explanation or {},
        created_at=row.created_at,
    )
