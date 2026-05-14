from __future__ import annotations

import asyncio
import json
import traceback
from pathlib import Path
from typing import Any
from uuid import UUID, uuid4

from fastapi import APIRouter, BackgroundTasks, Body, Depends, File, Form, HTTPException, UploadFile
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session, joinedload

from app.core.config import settings
from app.core.database import get_db
from app.models.evaluation import Evaluation as EvaluationModel
from app.models.evaluation_feedback import EvaluationFeedback
from app.models.attempt import Attempt
from app.schemas.evaluation_schema import (
    AttemptProgressOut,
    EvaluationHistoryOut,
    EvaluationMetrics,
    EvaluationResultOut,
    PersistedEvaluationOut,
)
from app.schemas.upload_schema import EvaluationStatusResponse
from app.services.evaluation.orchestrator import (
    run_evaluation_pipeline as _run_new_pipeline,
)
from app.services.progress_service import compute_progress

router = APIRouter(prefix="/api/evaluations", tags=["Evaluations"])

# ── SSE progress store ─────────────────────────────────────────────────────────
# Keyed by evaluation_id (str). Values are replaced in-place as steps complete.
_evaluation_progress: dict[str, dict] = {}

PROGRESS_STEPS: dict[str, dict] = {
    "upload":            {"step": "upload",            "label": "Uploading your video",             "progress": 5},
    "yolo":              {"step": "yolo",              "label": "Detecting scissors in your video", "progress": 15},
    "trajectory_init":   {"step": "trajectory_init",   "label": "Initializing trajectory tracking", "progress": 25},
    "trajectory_track":  {"step": "trajectory_track",  "label": "Tracking scissor path",            "progress": 40},
    "trajectory_errors": {"step": "trajectory_errors", "label": "Detecting trajectory errors",      "progress": 55},
    "angle_init":        {"step": "angle_init",        "label": "Analyzing cutting angles",         "progress": 68},
    "angle_track":       {"step": "angle_track",       "label": "Comparing angles to expert",       "progress": 82},
    "angle_errors":      {"step": "angle_errors",      "label": "Detecting angle errors",           "progress": 92},
    "done":              {"step": "done",               "label": "Analysis complete",                "progress": 100},
}


def _emit_progress(evaluation_id: str, step: str) -> None:
    if step in PROGRESS_STEPS:
        _evaluation_progress[evaluation_id] = dict(PROGRESS_STEPS[step])


def _run_orchestrator_bg(
    evaluation_id: str,
    learner_video_path: str,
    expert_id: str,
) -> None:
    def emit(step: str) -> None:
        if step != "done":
            _emit_progress(evaluation_id, step)

    try:
        result = _run_new_pipeline(
            learner_video_path=learner_video_path,
            expert_id=expert_id,
            emit_progress=emit,
        )
        _evaluation_progress[evaluation_id] = {
            **PROGRESS_STEPS["done"],
            "run_id": result.get("run_id"),
        }
        print(f"[EVAL] Background orchestrator complete — run_id: {result.get('run_id')}")
    except Exception as exc:
        _evaluation_progress[evaluation_id] = {
            "step": "error",
            "label": f"Evaluation failed: {exc}",
            "progress": -1,
        }
        print(f"[EVAL] Background orchestrator FAILED: {exc}")
        traceback.print_exc()


# ── POST /start ────────────────────────────────────────────────────────────────

@router.post("/start")
async def start_evaluation(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    clip_id: str = Form(...),
):
    # 1. Save uploaded file to disk
    suffix = Path(file.filename or "upload.mp4").suffix or ".mp4"
    save_path = settings.STORAGE_ROOT / "uploads" / "compare_tmp" / f"{uuid4().hex}{suffix}"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, "wb") as f:
        f.write(await file.read())

    # 2. Generate evaluation ID
    evaluation_id = str(uuid4())

    # 3. Seed initial progress so SSE has something to return immediately
    _emit_progress(evaluation_id, "upload")

    # 4. Run pipeline in background
    background_tasks.add_task(
        _run_orchestrator_bg,
        evaluation_id,
        str(save_path),
        clip_id,
    )

    print(f"[EVAL] Orchestrator started — evaluation_id: {evaluation_id}, expert_id: {clip_id}")

    # 5. Return immediately
    return {"evaluation_id": evaluation_id, "status": "processing"}


# ── On-demand corridor overlay generation ─────────────────────────────────────

@router.post("/{evaluation_id}/generate-corridor-overlay")
async def generate_corridor_overlay(
    evaluation_id: str,
    body: dict = Body(...),
):
    """Generate the corridor overlay video on demand for a completed evaluation run.

    Body: { "run_id": "<uuid>" }
    Returns: { "status": "done", "overlay_video_url": "/storage/..." }
    """
    run_id: str | None = body.get("run_id")
    if not run_id:
        raise HTTPException(status_code=400, detail="run_id is required in the request body.")

    eval_base_dir = settings.STORAGE_ROOT / "evaluation"
    run_dir = eval_base_dir / run_id
    trajectory_dir = run_dir / "trajectory"

    print(f"[PATH OVERLAY] run_id received: {run_id}")
    print(f"[PATH OVERLAY] looking for dir: {eval_base_dir / run_id}")
    print(f"[PATH OVERLAY] dir exists: {(eval_base_dir / run_id).exists()}")

    if not (trajectory_dir / "aligned_corridor.json").is_file():
        raise HTTPException(
            status_code=404,
            detail="Corridor data not found for this run. Run evaluation first.",
        )

    try:
        from app.services.sam2_yolo.pipeline import generate_corridor_overlay_for_run  # noqa: PLC0415

        result = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: generate_corridor_overlay_for_run(
                run_id=run_id,
                eval_base_dir=str(eval_base_dir),
            ),
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Overlay generation failed: {exc}",
        ) from exc

    overlay_url: str | None = result.get("overlay_video_url")
    if not overlay_url:
        raise HTTPException(status_code=500, detail="Overlay video URL not returned by pipeline.")

    return {"status": "done", "overlay_video_url": overlay_url}


# ── On-demand visualization generation ────────────────────────────────────────

@router.post("/{evaluation_id}/generate-visualization")
async def generate_visualization(
    evaluation_id: str,
    body: dict = Body(...),
):
    """Generate the mask-based visualization video on demand for a completed evaluation run.

    Body: { "run_id": "<uuid>", "expert_id": "<uuid>" }
    Returns: { "status": "done", "visualization_url": "/storage/evaluation/{run_id}/visualization/visualization.mp4" }
    """
    run_id: str | None = body.get("run_id")
    expert_id: str | None = body.get("expert_id")
    if not run_id:
        raise HTTPException(status_code=400, detail="run_id is required in the request body.")
    if not expert_id:
        raise HTTPException(status_code=400, detail="expert_id is required in the request body.")

    eval_base_dir = settings.STORAGE_ROOT / "evaluation"
    run_dir = eval_base_dir / run_id

    yolo_path = run_dir / "yolo_detections.json"
    if not yolo_path.is_file():
        raise HTTPException(status_code=404, detail="yolo_detections.json not found for this run.")

    with open(yolo_path) as fh:
        yolo_data = json.load(fh)
    learner_video_path: str | None = yolo_data.get("video_path")
    if not learner_video_path or not Path(learner_video_path).is_file():
        raise HTTPException(status_code=404, detail="Learner video path not found or file missing.")

    output_dir = run_dir / "visualization"

    try:
        from app.services.evaluation.visualization import render_visualization  # noqa: PLC0415

        out_path = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: render_visualization(
                learner_video_path=learner_video_path,
                expert_id=expert_id,
                run_id=run_id,
                output_dir=str(output_dir),
            ),
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Visualization generation failed: {exc}",
        ) from exc

    visualization_url = f"/storage/evaluation/{run_id}/visualization/visualization.mp4"
    return {"status": "done", "visualization_url": visualization_url}


# ── On-demand VLM feedback generation ────────────────────────────────────────

@router.post("/{evaluation_id}/generate-feedback")
async def generate_feedback(
    evaluation_id: str,
    body: dict = Body(...),
):
    """Generate VLM coaching feedback for a completed evaluation run.

    Body: { "run_id": "<uuid>", "expert_id": "<uuid>" }
    Returns: { "status": "done", "feedback": "<text>", "feedback_url": "/storage/..." }

    If feedback.json already exists for this run it is returned without calling the API.
    """
    run_id: str | None = body.get("run_id")
    expert_id: str | None = body.get("expert_id")
    if not run_id:
        raise HTTPException(status_code=400, detail="run_id is required in the request body.")
    if not expert_id:
        raise HTTPException(status_code=400, detail="expert_id is required in the request body.")

    feedback_path = (
        Path(settings.STORAGE_ROOT) / "evaluation" / run_id / "vlm" / "feedback.json"
    )

    if feedback_path.exists():
        with open(feedback_path, encoding="utf-8") as fh:
            cached = json.load(fh)
        return {
            "status": "done",
            "feedback": cached.get("feedback", ""),
            "feedback_url": f"/storage/evaluation/{run_id}/vlm/feedback.json",
        }

    output_dir = str(Path(settings.STORAGE_ROOT) / "evaluation" / run_id / "visualization")

    try:
        from app.services.evaluation.vlm_feedback import generate_feedback as _generate_feedback  # noqa: PLC0415

        saved_path = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: _generate_feedback(
                run_id=run_id,
                expert_id=expert_id,
                output_dir=output_dir,
            ),
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Feedback generation failed: {exc}",
        ) from exc

    with open(saved_path, encoding="utf-8") as fh:
        result = json.load(fh)

    return {
        "status": "done",
        "feedback": result.get("feedback", ""),
        "feedback_url": f"/storage/evaluation/{run_id}/vlm/feedback.json",
    }


# ── SSE stream ─────────────────────────────────────────────────────────────────

@router.get("/{evaluation_id}/status-stream")
async def stream_evaluation_status(evaluation_id: str):
    """Server-Sent Events stream for evaluation progress."""
    async def event_generator():
        last_progress: int = -999
        elapsed = 0.0
        max_wait = 600.0  # 10 minutes

        while elapsed < max_wait:
            current = _evaluation_progress.get(evaluation_id)
            if current is not None:
                p = int(current.get("progress", 0))
                if p != last_progress:
                    last_progress = p
                    yield f"data: {json.dumps(current)}\n\n"
                if p >= 100 or p < 0:
                    break
            await asyncio.sleep(0.5)
            elapsed += 0.5

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )


# ── Unified errors ────────────────────────────────────────────────────────────

@router.get("/{evaluation_id}/errors")
async def get_unified_errors(evaluation_id: str, run_id: str):
    """Return the merged trajectory + angle error list for a completed run."""
    errors_path = (
        Path(settings.STORAGE_ROOT) / "evaluation" / run_id / "score" / "unified_errors.json"
    )
    if not errors_path.exists():
        raise HTTPException(status_code=404, detail="Unified errors not found for this run.")
    with open(errors_path) as fh:
        return json.load(fh)


# ── POST /{evaluation_id}/score ────────────────────────────────────────────────

NATIVE_W = 1920
NATIVE_H = 1080


def _scale_mark(mark: dict) -> tuple[float, float]:
    vw = mark.get("video_width") or NATIVE_W
    vh = mark.get("video_height") or NATIVE_H
    return mark["x"] * (NATIVE_W / vw), mark["y"] * (NATIVE_H / vh)


def _mark_hits_error(sx: float, sy: float, mark_type: str, error: dict) -> bool:
    if error["error_type"] != mark_type:
        return False
    bb = error.get("bounding_box")
    if bb is None:
        return False
    return bb["x_min"] <= sx <= bb["x_max"] and bb["y_min"] <= sy <= bb["y_max"]


def _run_feedback_bg(run_id: str, expert_id: str, output_dir: str) -> None:
    try:
        from app.services.evaluation.vlm_feedback import generate_feedback as _gen  # noqa: PLC0415
        _gen(run_id=run_id, expert_id=expert_id, output_dir=output_dir)
        print(f"[VLM] feedback generation complete for run {run_id}", flush=True)
    except Exception as exc:
        import traceback
        print(f"[VLM] FAILED for run {run_id}:", flush=True)
        traceback.print_exc()


@router.get("/{evaluation_id}/generate-feedback")
async def get_feedback_status(evaluation_id: str, run_id: str, expert_id: str = ""):
    """Poll whether VLM coaching feedback is ready for a completed run.

    Returns { "status": "pending" } while generation is in progress,
    or { "status": "done", "feedback": "...", "feedback_url": "..." } once ready.
    """
    feedback_path = (
        Path(settings.STORAGE_ROOT) / "evaluation" / run_id / "vlm" / "feedback.json"
    )
    if not feedback_path.exists():
        return {"status": "pending"}
    with open(feedback_path, encoding="utf-8") as fh:
        cached = json.load(fh)
    return {
        "status": "done",
        "feedback": cached.get("feedback", ""),
        "feedback_url": f"/storage/evaluation/{run_id}/vlm/feedback.json",
    }


@router.post("/{evaluation_id}/score")
async def score_evaluation(evaluation_id: str, background_tasks: BackgroundTasks, body: dict = Body(...)):
    """Score the learner's X-mark game attempt against the stored unified errors.

    Body: { "run_id": "...", "marks": [ { mark_type, x, y, timestamp_sec,
            video_width, video_height }, ... ] }
    """
    run_id: str | None = body.get("run_id")
    marks: list[dict] = body.get("marks", [])
    if not run_id:
        raise HTTPException(status_code=400, detail="run_id is required")

    errors_path = (
        Path(settings.STORAGE_ROOT) / "evaluation" / run_id / "score" / "unified_errors.json"
    )
    if not errors_path.exists():
        raise HTTPException(status_code=404, detail="unified_errors.json not found for this run")

    with open(errors_path) as fh:
        unified = json.load(fh)
    errors: list[dict] = unified.get("all_errors", [])

    matched_error_ids: set[int] = set()
    mark_results: list[dict] = []

    for mark in marks:
        sx, sy = _scale_mark(mark)
        hit_error: dict | None = None
        for error in errors:
            if error["error_id"] in matched_error_ids:
                continue
            if _mark_hits_error(sx, sy, mark["mark_type"], error):
                hit_error = error
                break

        if hit_error:
            matched_error_ids.add(hit_error["error_id"])
            mark_results.append({
                "mark_type": mark["mark_type"],
                "x": mark["x"],
                "y": mark["y"],
                "timestamp_sec": mark.get("timestamp_sec"),
                "result": "correct",
                "matched_error_id": hit_error["error_id"],
            })
        else:
            mark_results.append({
                "mark_type": mark["mark_type"],
                "x": mark["x"],
                "y": mark["y"],
                "timestamp_sec": mark.get("timestamp_sec"),
                "result": "false_alarm",
                "matched_error_id": None,
            })

    missed_errors = [e for e in errors if e["error_id"] not in matched_error_ids]
    total_real = len(errors)
    correct = sum(1 for m in mark_results if m["result"] == "correct")
    false_alarms = sum(1 for m in mark_results if m["result"] == "false_alarm")
    missed = len(missed_errors)
    score_pct = round(100 * correct / total_real) if total_real > 0 else 0

    result = {
        "run_id": run_id,
        "total_real_errors": total_real,
        "correct_marks": correct,
        "false_alarms": false_alarms,
        "missed_errors": missed,
        "score_pct": score_pct,
        "mark_results": mark_results,
        "missed_error_details": [
            {
                "error_id": e["error_id"],
                "error_type": e["error_type"],
                "timestamp_start_sec": e["timestamp_start_sec"],
                "timestamp_end_sec": e["timestamp_end_sec"],
                "peak_location": e.get("peak_location"),
            }
            for e in missed_errors
        ],
    }

    score_path = (
        Path(settings.STORAGE_ROOT) / "evaluation" / run_id / "score" / "score_result.json"
    )
    with open(score_path, "w") as fh:
        json.dump(result, fh, indent=2)

    # Kick off VLM feedback generation in the background (non-blocking).
    expert_id: str | None = unified.get("expert_id")
    if expert_id:
        feedback_file = (
            Path(settings.STORAGE_ROOT) / "evaluation" / run_id / "vlm" / "feedback.json"
        )
        if not feedback_file.exists():
            output_dir = str(
                Path(settings.STORAGE_ROOT) / "evaluation" / run_id / "visualization"
            )
            background_tasks.add_task(_run_feedback_bg, run_id, expert_id, output_dir)

    return result


# ── Remaining read-only DB endpoints (untouched) ──────────────────────────────

@router.get("/{evaluation_id}/status", response_model=EvaluationStatusResponse)
async def get_evaluation_status(evaluation_id: UUID, db: Session = Depends(get_db)):
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
    feedback = row.feedback if hasattr(row, "feedback") else None
    explanation_payload: dict[str, Any] = {}
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
