"""FastAPI routes for the YOLO + angle + DTW pipeline.

Endpoints:
  POST /api/angle/run-learner               — run full learner pipeline (returns run_id)
  GET  /api/angle/learner/{run_id}/result
  POST /api/angle/learner/{run_id}/generate-preview  — SYNC button only
  GET  /api/angle/expert/{expert_name}/status
"""

from __future__ import annotations

import json
import logging
import uuid
from pathlib import Path

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from pydantic import BaseModel

from app.core.config import settings
from app.services.angle.schemas import (
    ANGLES_FILENAME,
    DTW_ALIGNMENT_FILENAME,
    DTW_PREVIEW_FILENAME,
    RUN_SUMMARY_FILENAME,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/angle", tags=["Angle"])


# ── Request bodies ─────────────────────────────────────────────────────────────

class GeneratePreviewRequest(BaseModel):
    run_id: str


# ── POST /api/angle/run-learner ────────────────────────────────────────────────
# Accepts multipart/form-data: file (video), expert_name.
# Generates a fresh UUID4 run_id for every call — no overwriting.

@router.post("/run-learner")
async def run_learner(
    file: UploadFile = File(...),
    expert_name: str = Form(...),
) -> dict:
    from app.services.angle.learner_pipeline import run_learner_angle_pipeline

    expert_name = expert_name.strip()

    if not expert_name:
        raise HTTPException(status_code=400, detail="expert_name is required")
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename in upload")

    # Generate a unique run ID for this submission.
    run_id = str(uuid.uuid4())

    # Save uploaded video inside the new run folder.
    learner_out_dir = settings.ANGLES_OUTPUT_ROOT / "learner" / run_id
    learner_out_dir.mkdir(parents=True, exist_ok=True)

    suffix = Path(file.filename).suffix.lower()
    if suffix not in {".mp4", ".mov", ".avi", ".mkv", ".webm"}:
        suffix = ".mp4"

    video_path = learner_out_dir / f"input{suffix}"
    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="Uploaded file is empty")
    video_path.write_bytes(content)

    try:
        result = run_learner_angle_pipeline(
            video_path=video_path,
            run_id=run_id,
            expert_name=expert_name,
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("[angle] run_learner_angle_pipeline failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    summary_path = learner_out_dir / RUN_SUMMARY_FILENAME
    summary_data: dict = {}
    if summary_path.is_file():
        summary_data = json.loads(summary_path.read_text(encoding="utf-8"))

    return {
        "status": "done",
        "run_id": run_id,
        "dtw_path": str(learner_out_dir / DTW_ALIGNMENT_FILENAME),
        "learner_angles_path": str(learner_out_dir / ANGLES_FILENAME),
        "summary": {
            "dtw_distance": summary_data.get("dtw_distance"),
            "normalized_dtw_distance": summary_data.get("normalized_dtw_distance"),
            "mean_angle_difference": summary_data.get("mean_angle_difference"),
            "high_error_frame_count": summary_data.get("high_error_frame_count"),
            "medium_error_frame_count": summary_data.get("medium_error_frame_count"),
            "ok_frame_count": summary_data.get("ok_frame_count"),
        },
    }


# ── GET /api/angle/learner/{run_id}/result ─────────────────────────────────────

@router.get("/learner/{run_id}/result")
def get_learner_result(run_id: str) -> dict:
    dtw_path = settings.ANGLES_OUTPUT_ROOT / "learner" / run_id / DTW_ALIGNMENT_FILENAME

    if not dtw_path.is_file():
        raise HTTPException(
            status_code=404,
            detail=f"DTW result not found for run_id={run_id}. Run the pipeline first.",
        )

    return json.loads(dtw_path.read_text(encoding="utf-8"))


# ── POST /api/angle/learner/{run_id}/generate-preview ─────────────────────────

@router.post("/learner/{run_id}/generate-preview")
def generate_preview(run_id: str, body: GeneratePreviewRequest) -> dict:
    """Generate the DTW aligned preview video.

    Called ONLY when the user manually enables the SYNC toggle.
    This operation can take several minutes.
    """
    from app.services.angle.preview import create_dtw_aligned_preview

    learner_out_dir = settings.ANGLES_OUTPUT_ROOT / "learner" / run_id
    dtw_path = learner_out_dir / DTW_ALIGNMENT_FILENAME

    if not dtw_path.is_file():
        raise HTTPException(
            status_code=404,
            detail=f"DTW alignment not found for run_id={run_id}. Run the pipeline first.",
        )

    dtw_data = json.loads(dtw_path.read_text(encoding="utf-8"))
    dtw_matches = dtw_data.get("matches", [])
    if not dtw_matches:
        raise HTTPException(status_code=422, detail="DTW alignment has no matches")

    summary_path = learner_out_dir / RUN_SUMMARY_FILENAME
    if not summary_path.is_file():
        raise HTTPException(status_code=404, detail="run_summary.json not found")

    summary_data = json.loads(summary_path.read_text(encoding="utf-8"))
    expert_name = summary_data.get("expert_name", "")

    expert_angles_dir = settings.ANGLES_OUTPUT_ROOT / "expert" / expert_name
    if not expert_angles_dir.is_dir():
        raise HTTPException(
            status_code=404,
            detail=f"Expert output directory not found for expert_name={expert_name}",
        )

    preview_path = learner_out_dir / DTW_PREVIEW_FILENAME

    expert_video_path = expert_angles_dir / "annotated_expert.mp4"
    learner_video_path = learner_out_dir / "learner_comparison_output.mp4"

    for vpath, label in [
        (expert_video_path, "annotated_expert.mp4"),
        (learner_video_path, "learner_comparison_output.mp4"),
    ]:
        if not vpath.is_file():
            raise HTTPException(
                status_code=404,
                detail=(
                    f"{label} not found for run_id={run_id}. "
                    "Annotated videos must be generated before creating the preview."
                ),
            )

    try:
        create_dtw_aligned_preview(
            expert_video_path=expert_video_path,
            learner_video_path=learner_video_path,
            dtw_matches=dtw_matches,
            output_path=preview_path,
        )
    except Exception as exc:
        logger.exception("[angle] create_dtw_aligned_preview failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return {
        "status": "done",
        "run_id": run_id,
        "preview_path": str(preview_path),
    }


# ── GET /api/angle/expert/{expert_name}/status ────────────────────────────────

@router.get("/expert/{expert_name}/status")
def get_expert_status(expert_name: str) -> dict:
    angles_path = (
        settings.ANGLES_OUTPUT_ROOT / "expert" / expert_name / ANGLES_FILENAME
    )
    return {
        "exists": angles_path.is_file(),
        "expert_name": expert_name,
    }
