from __future__ import annotations

import hashlib
import traceback
from pathlib import Path
from typing import Any
from uuid import UUID, uuid4

from fastapi import APIRouter, Depends, HTTPException, Request
from sqlalchemy.orm import Session

from app.core.config import settings
from app.core.database import get_db
from app.models.evaluation import Evaluation as EvaluationModel
from app.models.video import Video
from app.models.attempt import Attempt
from app.schemas.upload_schema import (
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
from app.services.context_gate_service import run_context_gate, OUT_OF_CONTEXT_MESSAGE
from app.services.evaluation_engine_service import run_evaluation_pipeline
from app.services.media_service import find_first_expert_video_file, normalize_storage_key
from app.services.motion_representation_service import process_video_to_motion_representation
from app.services.progress_service import compute_progress

router = APIRouter(prefix="/api/evaluations", tags=["Evaluations"])

MIN_VALID_LANDMARK_FRAMES = 3


@router.post("/start", response_model=EvaluationStartResponse)
async def start_evaluation(request: Request, db: Session = Depends(get_db)):
    """Run the real evaluation pipeline: video -> MediaPipe -> motion -> DTW -> score."""

    raw_attempt_id = None
    selected_course_id = None
    selected_clip_id = None
    selected_chapter_id = None
    learner_video_path: Path | None = None
    content_type = (request.headers.get("content-type") or "").lower()
    if "multipart/form-data" in content_type:
        form = await request.form()
        raw_attempt_id = form.get("attempt_id")
        selected_course_id = form.get("course_id")
        selected_clip_id = form.get("clip_id")
        selected_chapter_id = form.get("chapter_id")
        uploaded_file = form.get("file")
        if uploaded_file is not None and hasattr(uploaded_file, "filename"):
            learner_video_path = await _save_uploaded_file(uploaded_file)
    else:
        try:
            payload = await request.json()
        except Exception:
            payload = {}
        if isinstance(payload, dict):
            raw_attempt_id = payload.get("attempt_id")
            selected_course_id = payload.get("course_id")
            selected_clip_id = payload.get("clip_id")
            selected_chapter_id = payload.get("chapter_id")

    attempt_id = _coerce_uuid(raw_attempt_id)
    chapter_id = _resolve_chapter_id(db=db, attempt_id=attempt_id, chapter_id_value=selected_chapter_id)
    expert_video_path = _resolve_expert_video_path(
        db=db,
        chapter_id=chapter_id,
        course_id=str(selected_course_id) if selected_course_id else None,
        clip_id=str(selected_clip_id) if selected_clip_id else None,
    )

    print(
        "[EVAL] INPUTS:",
        {
            "mode": "REAL_MEDIAPIPE",
            "course_id": selected_course_id,
            "clip_id": selected_clip_id,
            "chapter_id": str(chapter_id) if chapter_id else None,
            "expert_video_path": str(expert_video_path) if expert_video_path else None,
            "learner_video_path": str(learner_video_path) if learner_video_path else None,
        },
    )

    if expert_video_path is None or not expert_video_path.exists():
        raise HTTPException(
            status_code=400,
            detail=f"Expert video not found. Resolved path: {expert_video_path}",
        )
    if learner_video_path is None or not learner_video_path.exists():
        raise HTTPException(
            status_code=400,
            detail="Learner video not uploaded or file missing.",
        )

    try:
        expert_motion_output = process_video_to_motion_representation(str(expert_video_path))
        expert_valid = _count_valid_landmark_frames(expert_motion_output)
        print(f"[EVAL] Expert landmarks: total_frames={expert_motion_output.total_frames}, valid_landmark_frames={expert_valid}")

        if expert_valid < MIN_VALID_LANDMARK_FRAMES:
            raise HTTPException(
                status_code=422,
                detail=f"Expert video has insufficient hand landmarks ({expert_valid} valid frames, need >= {MIN_VALID_LANDMARK_FRAMES}).",
            )
    except HTTPException:
        raise
    except Exception as exc:
        print(f"[EVAL] Expert landmark extraction FAILED: {exc}")
        traceback.print_exc()
        raise HTTPException(
            status_code=422,
            detail=f"Failed to extract landmarks from expert video: {exc}",
        )

    try:
        learner_motion_output = process_video_to_motion_representation(str(learner_video_path))
        learner_valid = _count_valid_landmark_frames(learner_motion_output)
        print(f"[EVAL] Learner landmarks: total_frames={learner_motion_output.total_frames}, valid_landmark_frames={learner_valid}")

        if learner_valid < MIN_VALID_LANDMARK_FRAMES:
            raise HTTPException(
                status_code=422,
                detail=f"Learner video has insufficient hand landmarks ({learner_valid} valid frames, need >= {MIN_VALID_LANDMARK_FRAMES}). Make sure hands are visible in the video.",
            )
    except HTTPException:
        raise
    except Exception as exc:
        print(f"[EVAL] Learner landmark extraction FAILED: {exc}")
        traceback.print_exc()
        raise HTTPException(
            status_code=422,
            detail=f"Failed to extract landmarks from learner video: {exc}",
        )

    expert_motion_dict = expert_motion_output.model_dump()
    learner_motion_dict = learner_motion_output.model_dump()

    print(
        "[EVAL] Motion sequence lengths:",
        {
            "expert_sequence_length": expert_motion_dict.get("sequence_length"),
            "learner_sequence_length": learner_motion_dict.get("sequence_length"),
        },
    )

    # ── Context gate: reject clearly out-of-context videos early ──────
    gate_result = run_context_gate(
        expert_motion=expert_motion_dict,
        learner_motion=learner_motion_dict,
    )
    if not gate_result.passed:
        print(f"[EVAL] REJECTED by context gate: {gate_result.reasons}")
        return EvaluationStartResponse(
            evaluation_id=uuid4(),
            attempt_id=attempt_id,
            status="out_of_context",
            message=OUT_OF_CONTEXT_MESSAGE,
            score=0,
            gate_status="rejected",
            gate_reasons=gate_result.reasons,
        )

    evaluation_result = await run_evaluation_pipeline(
        expert_motion_source=expert_motion_dict,
        learner_motion_source=learner_motion_dict,
        attempt_id=str(attempt_id),
        db=db,
    )
    evaluation_id = UUID(str(evaluation_result.evaluation_id))
    print(
        "[EVAL] RESULT:",
        {
            "evaluation_id": str(evaluation_id),
            "score": evaluation_result.score,
            "angle_deviation": evaluation_result.metrics.angle_deviation,
            "trajectory_deviation": evaluation_result.metrics.trajectory_deviation,
            "velocity_difference": evaluation_result.metrics.velocity_difference,
            "smoothness_score": evaluation_result.metrics.smoothness_score,
            "timing_score": evaluation_result.metrics.timing_score,
        },
    )

    return EvaluationStartResponse(
        evaluation_id=evaluation_id,
        attempt_id=attempt_id,
        status="completed",
        message="Evaluation completed.",
    )


@router.get("/{evaluation_id}/status", response_model=EvaluationStatusResponse)
async def get_evaluation_status(evaluation_id: UUID):
    """Poll evaluation processing status."""
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
        db.query(EvaluationModel)
        .filter(EvaluationModel.attempt_id == attempt_id)
        .order_by(EvaluationModel.created_at.desc())
        .all()
    )
    return EvaluationHistoryOut(evaluations=[_to_persisted_out(row) for row in rows])


@router.get("/progress/{attempt_id}", response_model=AttemptProgressOut)
async def get_evaluation_progress(attempt_id: str, db: Session = Depends(get_db)):
    rows = (
        db.query(EvaluationModel)
        .filter(EvaluationModel.attempt_id == attempt_id)
        .order_by(EvaluationModel.created_at.desc())
        .all()
    )
    evaluations = [_to_persisted_out(row).model_dump() for row in rows]
    computed = compute_progress(evaluations)
    return AttemptProgressOut(attempt_id=attempt_id, **computed)


@router.get("/{evaluation_id}", response_model=PersistedEvaluationOut)
async def get_evaluation_by_id(evaluation_id: str, db: Session = Depends(get_db)):
    row = db.query(EvaluationModel).filter(EvaluationModel.id == evaluation_id).first()
    if row is None:
        raise HTTPException(status_code=404, detail="Evaluation not found")
    return _to_persisted_out(row)


def _to_persisted_out(row: EvaluationModel) -> PersistedEvaluationOut:
    return PersistedEvaluationOut(
        id=str(row.id),
        attempt_id=str(row.attempt_id),
        score=float(row.overall_score or 0.0),
        metrics=row.metrics or {},
        per_metric_breakdown=row.per_metric_breakdown or {},
        key_error_moments=row.key_error_moments or [],
        semantic_phases=row.semantic_phases or {},
        explanation={},
        created_at=row.created_at,
    )


def _coerce_uuid(value) -> UUID:
    try:
        return UUID(str(value))
    except Exception:
        return uuid4()


def _resolve_chapter_id(db: Session, attempt_id: UUID, chapter_id_value: Any) -> UUID | None:
    try:
        if chapter_id_value:
            return UUID(str(chapter_id_value))
    except Exception:
        pass

    attempt = db.query(Attempt).filter(Attempt.id == str(attempt_id)).first()
    if attempt is None:
        return None
    try:
        return UUID(str(attempt.chapter_id))
    except Exception:
        return None


def _resolve_expert_video_path(
    *,
    db: Session,
    chapter_id: UUID | None,
    course_id: str | None,
    clip_id: str | None,
) -> Path | None:
    if chapter_id is not None:
        row = db.query(Video).filter(Video.chapter_id == str(chapter_id), Video.video_role == "expert").first()
        if row is not None and row.file_path:
            try:
                return settings.STORAGE_ROOT / normalize_storage_key(row.file_path)
            except Exception:
                pass

    if course_id == "pottery-wheel" and clip_id == "pottery-wheel-clip-1":
        mapped = settings.STORAGE_ROOT / "expert" / "pottery.mp4"
        if mapped.exists():
            return mapped

    if course_id == "pottery-wheel" and clip_id == "pottery-wheel-clip-2":
        mapped = settings.STORAGE_ROOT / "expert" / "expert1.mp4"
        if mapped.exists():
            return mapped

    return find_first_expert_video_file()


async def _save_uploaded_file(uploaded_file) -> Path:
    suffix = Path(str(uploaded_file.filename or "upload.mp4")).suffix or ".mp4"
    relative = Path("uploads") / "compare_tmp" / f"{uuid4().hex}{suffix}"
    destination = settings.STORAGE_ROOT / relative
    destination.parent.mkdir(parents=True, exist_ok=True)

    with destination.open("wb") as output:
        while True:
            chunk = await uploaded_file.read(1024 * 1024)
            if not chunk:
                break
            output.write(chunk)
    await uploaded_file.close()

    return destination


def _count_valid_landmark_frames(motion_output) -> int:
    count = 0
    for frame in motion_output.frames:
        left = frame.left_hand_features
        right = frame.right_hand_features
        if left.present or right.present:
            count += 1
    return count
