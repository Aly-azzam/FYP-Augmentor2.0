from __future__ import annotations

import traceback
from pathlib import Path
from typing import Any
from uuid import UUID, uuid4

from fastapi import APIRouter, Depends, HTTPException, Request
from sqlalchemy.orm import Session, joinedload

from app.core.config import settings
from app.core.constants import AttemptStatus
from app.core.database import get_db
from app.models.evaluation import Evaluation as EvaluationModel
from app.models.evaluation_feedback import EvaluationFeedback
from app.models.video import Video
from app.models.attempt import Attempt
from app.models.user import User
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
from app.services.optical_flow import (
    FarnebackConfig,
    evaluate_optical_flow_summary,
    run_optical_flow_comparison,
)
from app.services.optical_flow.io_utils import save_optical_flow_results
from app.services.optical_flow.schemas import (
    OpticalFlowEvaluationConfigUsed,
    OpticalFlowEvaluationResult,
    OpticalFlowScore,
    OpticalFlowSimilarities,
)
from app.services.optical_flow.visualizer import create_comparison_visualizations
from app.services.progress_service import compute_progress

router = APIRouter(prefix="/api/evaluations", tags=["Evaluations"])

MIN_VALID_LANDMARK_FRAMES = 3


def _build_schema_evaluation_result(evaluation_result: dict) -> OpticalFlowEvaluationResult:
    return OpticalFlowEvaluationResult(
        similarities=OpticalFlowSimilarities(**evaluation_result["similarities"]),
        score=OpticalFlowScore(**evaluation_result["score"]),
        config_used=OpticalFlowEvaluationConfigUsed(**evaluation_result["config_used"]),
    )


def _run_optical_flow_side_analysis(
    *,
    expert_video_path: Path,
    learner_video_path: Path,
    attempt_id: UUID,
) -> dict[str, Any]:
    output_dir = settings.STORAGE_ROOT / "outputs" / "optical_flow" / str(attempt_id)
    visualization_dir = output_dir / "visualizations"
    config = FarnebackConfig(
        use_hand_roi=True,
        roi_padding_px=40,
    )

    raw_result, summary_result = run_optical_flow_comparison(
        expert_video_path=expert_video_path,
        learner_video_path=learner_video_path,
        config=config,
    )
    evaluation_result = evaluate_optical_flow_summary(summary_result)
    summary_result.optical_flow_evaluation = _build_schema_evaluation_result(
        evaluation_result
    )

    raw_json_path, summary_json_path = save_optical_flow_results(
        raw_result=raw_result,
        summary_result=summary_result,
        output_dir=output_dir,
    )
    expert_hsv_video_path, learner_hsv_video_path = create_comparison_visualizations(
        expert_video_path=expert_video_path,
        learner_video_path=learner_video_path,
        output_dir=visualization_dir,
        run_id=raw_result.run.run_id,
        config=config,
    )

    comparison_metrics = summary_result.comparison_metrics
    return {
        "run_id": raw_result.run.run_id,
        "raw_json_path": str(raw_json_path),
        "summary_json_path": str(summary_json_path),
        "expert_hsv_video_path": str(expert_hsv_video_path),
        "learner_hsv_video_path": str(learner_hsv_video_path),
        "optical_flow_score": evaluation_result["score"]["optical_flow_score"],
        "vibration_difference": comparison_metrics.vibration_difference,
        "vibration_ratio": comparison_metrics.vibration_ratio,
        "expert_vibration_score": summary_result.expert_summary.vibration_score,
        "learner_vibration_score": summary_result.learner_summary.vibration_score,
        "expert_roi_usage_ratio": summary_result.expert_summary.roi_usage_ratio,
        "learner_roi_usage_ratio": summary_result.learner_summary.roi_usage_ratio,
    }


@router.post("/start", response_model=EvaluationStartResponse)
async def start_evaluation(
    request: Request,
    db: Session = Depends(get_db),
):
    """Run the real evaluation pipeline: video -> MediaPipe -> motion -> DTW -> score."""

    raw_attempt_id = None
    selected_course_id = None
    selected_clip_id = None
    selected_chapter_id = None
    learner_video_path: Path | None = None
    uploaded_filename: str | None = None
    uploaded_content_type: str | None = None

    content_type = (request.headers.get("content-type") or "").lower()
    if "multipart/form-data" in content_type:
        form = await request.form()
        raw_attempt_id = form.get("attempt_id") or raw_attempt_id
        selected_course_id = form.get("course_id")
        selected_clip_id = form.get("clip_id")
        selected_chapter_id = form.get("chapter_id")
        uploaded_file = form.get("file")
        if uploaded_file is not None and hasattr(uploaded_file, "filename"):
            uploaded_filename = str(getattr(uploaded_file, "filename", "") or "")
            uploaded_content_type = str(getattr(uploaded_file, "content_type", "") or "") or None
            learner_video_path = await _save_uploaded_file(uploaded_file)
    else:
        try:
            payload = await request.json()
        except Exception:
            payload = {}
        if isinstance(payload, dict):
            raw_attempt_id = payload.get("attempt_id") or raw_attempt_id
            selected_course_id = payload.get("course_id")
            selected_clip_id = payload.get("clip_id")
            selected_chapter_id = payload.get("chapter_id")

    attempt_id: UUID | None = None
    if raw_attempt_id is not None and str(raw_attempt_id).strip() != "":
        attempt_id = _require_uuid(raw_attempt_id, field_name="attempt_id")

    if attempt_id is None:
        chapter_id = _resolve_optional_chapter_id(
            chapter_id_value=selected_chapter_id,
            clip_id_value=selected_clip_id,
        )
        if chapter_id is None:
            raise HTTPException(
                status_code=400,
                detail="attempt_id is required unless chapter_id (or a UUID clip_id) is provided with a learner file.",
            )
        if learner_video_path is None:
            raise HTTPException(
                status_code=400,
                detail="Learner video file is required when attempt_id is not provided.",
            )
        attempt_id = _create_temporary_attempt(
            db=db,
            chapter_id=chapter_id,
            learner_video_path=learner_video_path,
            original_filename=uploaded_filename,
            mime_type=uploaded_content_type,
        )
    else:
        chapter_id = _resolve_chapter_id(db=db, attempt_id=attempt_id, chapter_id_value=selected_chapter_id)

    if chapter_id is None:
        raise HTTPException(status_code=400, detail="Unable to resolve chapter_id for the provided attempt_id.")

    if learner_video_path is None:
        learner_video_path = _resolve_learner_video_path(db=db, attempt_id=attempt_id)

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

    optical_flow_side_analysis = None
    try:
        optical_flow_side_analysis = _run_optical_flow_side_analysis(
            expert_video_path=expert_video_path,
            learner_video_path=learner_video_path,
            attempt_id=attempt_id,
        )
        print("[OPTICAL_FLOW] SIDE ANALYSIS:", optical_flow_side_analysis)
    except Exception as exc:
        print(f"[OPTICAL_FLOW] Side analysis failed but evaluation will continue: {exc}")
        traceback.print_exc()

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
        persisted_evaluation_id = _persist_out_of_context_result(
            db=db,
            attempt_id=attempt_id,
            gate_reasons=gate_result.reasons,
        )
        print(f"[EVAL] REJECTED by context gate: {gate_result.reasons}")
        return EvaluationStartResponse(
            evaluation_id=UUID(persisted_evaluation_id),
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
    if optical_flow_side_analysis is not None:
        try:
            evaluation_row = (
                db.query(EvaluationModel)
                .filter(EvaluationModel.id == str(evaluation_id))
                .first()
            )
            if evaluation_row is not None:
                metrics = dict(evaluation_row.metrics or {})
                metrics["optical_flow"] = optical_flow_side_analysis
                evaluation_row.metrics = metrics
                db.commit()
            else:
                print(
                    "[OPTICAL_FLOW] Could not attach side analysis: "
                    f"evaluation row not found for {evaluation_id}"
                )
        except Exception as exc:
            db.rollback()
            print(
                "[OPTICAL_FLOW] Failed to persist side analysis but evaluation "
                f"will continue: {exc}"
            )
            traceback.print_exc()

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
async def get_evaluation_status(evaluation_id: UUID, db: Session = Depends(get_db)):
    """Poll evaluation processing status."""
    row = (
        db.query(EvaluationModel)
        .filter(EvaluationModel.id == str(evaluation_id))
        .first()
    )
    if row is None:
        raise HTTPException(status_code=404, detail="Evaluation not found")

    return EvaluationStatusResponse(
        evaluation_id=evaluation_id,
        attempt_id=UUID(str(row.attempt_id)),
        status=str(row.status),
        current_stage="completed" if str(row.status) in {"completed", "evaluated"} else "processing",
        message="Evaluation completed." if str(row.status) in {"completed", "evaluated"} else "Evaluation in progress.",
    )


@router.get("/{evaluation_id}/result", response_model=EvaluationResultOut)
async def get_evaluation_result(evaluation_id: UUID, db: Session = Depends(get_db)):
    """Get final evaluation result. Only available when status is terminal."""
    row = (
        db.query(EvaluationModel)
        .options(
            joinedload(EvaluationModel.attempt),
            joinedload(EvaluationModel.feedback),
        )
        .filter(EvaluationModel.id == str(evaluation_id))
        .first()
    )
    if row is None:
        raise HTTPException(status_code=404, detail="Evaluation not found")

    metrics_payload = row.metrics or {}
    metrics = EvaluationMetrics(
        angle_deviation=float(metrics_payload.get("angle_deviation", 0.0)),
        trajectory_deviation=float(metrics_payload.get("trajectory_deviation", 0.0)),
        velocity_difference=float(metrics_payload.get("velocity_difference", 0.0)),
        smoothness_score=float(metrics_payload.get("smoothness_score", 0.0)),
        timing_score=float(metrics_payload.get("timing_score", 0.0)),
        hand_openness_deviation=float(metrics_payload.get("hand_openness_deviation", 0.0)),
        tool_alignment_deviation=float(metrics_payload.get("tool_alignment_deviation", 0.0)),
        dtw_similarity=float(metrics_payload.get("dtw_similarity", 0.0)),
    )

    chapter_id = None
    if row.attempt is not None:
        try:
            chapter_id = UUID(str(row.attempt.chapter_id))
        except Exception:
            chapter_id = None
    if chapter_id is None:
        raise HTTPException(status_code=500, detail="Evaluation data is missing chapter linkage.")

    feedback_text = row.feedback.explanation_text if row.feedback is not None else None
    return EvaluationResultOut(
        evaluation_id=evaluation_id,
        chapter_id=chapter_id,
        expert_video_id=UUID(str(row.expert_video_id)),
        learner_video_id=UUID(str(row.learner_video_id)),
        status=str(row.status),
        score=int(round(float(row.overall_score or 0.0))),
        metrics=metrics,
        summary=row.summary_text,
        ai_text=feedback_text,
        created_at=row.created_at,
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
    feedback = (
        row.feedback
        if hasattr(row, "feedback")
        else None
    )
    explanation_payload = {}
    if feedback is not None:
        explanation_payload = {
            "mode": feedback.mode,
            "model_name": feedback.model_name,
            "explanation_text": feedback.explanation_text,
            "strengths": feedback.strengths or [],
            "weaknesses": feedback.weaknesses or [],
            "advice": feedback.advice,
            "cited_timestamps": feedback.cited_timestamps or [],
        }

    return PersistedEvaluationOut(
        id=str(row.id),
        attempt_id=str(row.attempt_id),
        score=float(row.overall_score or 0.0),
        metrics=row.metrics or {},
        per_metric_breakdown=row.per_metric_breakdown or {},
        key_error_moments=row.key_error_moments or [],
        semantic_phases=row.semantic_phases or {},
        explanation=explanation_payload,
        created_at=row.created_at,
    )


def _require_uuid(value: Any, field_name: str) -> UUID:
    if value is None or str(value).strip() == "":
        raise HTTPException(status_code=400, detail=f"{field_name} is required.")
    try:
        return UUID(str(value))
    except Exception:
        raise HTTPException(status_code=400, detail=f"Invalid {field_name}. Expected UUID format.")


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


def _resolve_optional_chapter_id(*, chapter_id_value: Any, clip_id_value: Any) -> UUID | None:
    """Resolve chapter UUID directly from request fields without requiring attempt_id."""
    for value in (chapter_id_value, clip_id_value):
        if value is None or str(value).strip() == "":
            continue
        try:
            return UUID(str(value))
        except Exception:
            continue
    return None


def _create_temporary_attempt(
    *,
    db: Session,
    chapter_id: UUID,
    learner_video_path: Path,
    original_filename: str | None,
    mime_type: str | None,
) -> UUID:
    """Create a lightweight attempt+learner video record for ad-hoc Compare Studio uploads."""
    user = db.query(User).order_by(User.created_at.asc()).first()
    if user is None:
        raise HTTPException(
            status_code=400,
            detail="No user record found to own this attempt. Seed at least one user first.",
        )

    try:
        learner_video = Video(
            owner_user_id=user.id,
            chapter_id=str(chapter_id),
            video_role="learner",
            file_path=normalize_storage_key(learner_video_path),
            file_name=original_filename or learner_video_path.name,
            mime_type=mime_type,
            file_size_bytes=learner_video_path.stat().st_size if learner_video_path.exists() else None,
        )
        db.add(learner_video)
        db.flush()

        attempt = Attempt(
            user_id=user.id,
            chapter_id=str(chapter_id),
            learner_video_id=learner_video.id,
            status=AttemptStatus.UPLOADED.value,
            original_filename=original_filename or learner_video_path.name,
        )
        db.add(attempt)
        db.commit()
        db.refresh(attempt)
        return UUID(str(attempt.id))
    except HTTPException:
        db.rollback()
        raise
    except Exception as exc:
        db.rollback()
        raise HTTPException(
            status_code=500,
            detail=f"Failed to create temporary learner attempt: {exc}",
        ) from exc


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


def _resolve_learner_video_path(*, db: Session, attempt_id: UUID) -> Path | None:
    attempt = (
        db.query(Attempt)
        .options(joinedload(Attempt.learner_video))
        .filter(Attempt.id == str(attempt_id))
        .first()
    )
    if attempt is None:
        raise HTTPException(status_code=404, detail="Attempt not found for provided attempt_id.")
    if attempt.learner_video is None or not attempt.learner_video.file_path:
        raise HTTPException(status_code=400, detail="Learner video is not linked to the provided attempt.")

    try:
        resolved_path = settings.STORAGE_ROOT / normalize_storage_key(attempt.learner_video.file_path)
    except Exception:
        raise HTTPException(status_code=400, detail="Learner video file path is invalid.")

    if not resolved_path.exists():
        raise HTTPException(status_code=400, detail="Learner video file does not exist on disk.")
    return resolved_path


def _persist_out_of_context_result(*, db: Session, attempt_id: UUID, gate_reasons: list[str]) -> str:
    attempt = db.query(Attempt).filter(Attempt.id == str(attempt_id)).first()
    if attempt is None:
        raise HTTPException(status_code=404, detail="Attempt not found for provided attempt_id.")

    expert_video = (
        db.query(Video)
        .filter(Video.chapter_id == attempt.chapter_id, Video.video_role == "expert")
        .order_by(Video.created_at.asc())
        .first()
    )
    if expert_video is None:
        raise HTTPException(
            status_code=400,
            detail="Expert video record not found for this chapter; cannot persist out_of_context result.",
        )

    row = db.query(EvaluationModel).filter(EvaluationModel.attempt_id == attempt.id).first()
    if row is None:
        row = EvaluationModel(
            attempt_id=attempt.id,
            expert_video_id=expert_video.id,
            learner_video_id=attempt.learner_video_id,
        )
        db.add(row)

    row.overall_score = 0
    row.status = "out_of_context"
    row.gate_passed = False
    row.gate_reasons = gate_reasons
    row.metrics = {}
    row.per_metric_breakdown = {}
    row.key_error_moments = []
    row.semantic_phases = {}
    row.summary_text = OUT_OF_CONTEXT_MESSAGE
    attempt.status = "out_of_context"

    feedback_text = OUT_OF_CONTEXT_MESSAGE
    if gate_reasons:
        feedback_text = f"{OUT_OF_CONTEXT_MESSAGE} Reasons: " + "; ".join(gate_reasons)

    feedback_row = (
        db.query(EvaluationFeedback)
        .filter(EvaluationFeedback.evaluation_id == row.id)
        .first()
    )
    if feedback_row is None:
        feedback_row = EvaluationFeedback(
            evaluation_id=row.id,
            mode="rule_based",
            explanation_text=feedback_text,
        )
        db.add(feedback_row)

    feedback_row.mode = "rule_based"
    feedback_row.model_name = "context_gate"
    feedback_row.explanation_text = feedback_text
    feedback_row.strengths = None
    feedback_row.weaknesses = gate_reasons or None
    feedback_row.advice = [
        "Please upload a learner practice video that matches the same chapter task as the expert."
    ]
    feedback_row.cited_timestamps = None

    db.commit()
    db.refresh(row)
    return str(row.id)


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
