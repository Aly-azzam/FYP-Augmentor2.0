from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import Any
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Request, status
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session, joinedload
from starlette.datastructures import UploadFile as StarletteUploadFile

from app.core.config import settings
from app.core.database import get_db
from app.models.attempt import Attempt
from app.models.video import Video
from app.services.media_service import build_storage_url, normalize_storage_key
from app.services.sam2_yolo.runner import DEFAULT_FRAME_STRIDE, run_sam2_yolo_scissors_tracking
from app.services.sam2_yolo.schemas import (
    METRICS_FILENAME,
    MODEL_NAME,
    OVERLAY_FILENAME,
    RAW_FILENAME,
    SUMMARY_FILENAME,
    TRACKING_POINT_TYPE,
)
from app.services.upload_service import UploadValidationError, save_upload_file, validate_upload_file


router = APIRouter(prefix="/api/sam2-yolo", tags=["SAM2 YOLO"])

SAM2_YOLO_RUNS_ROOT = Path(settings.STORAGE_ROOT) / "outputs" / "sam2_yolo" / "runs"
SAM2_YOLO_EXPERTS_ROOT = Path(settings.STORAGE_ROOT) / "outputs" / "sam2_yolo" / "experts"
EXPERT_MISSING_WARNING = "Precomputed expert SAM2+YOLO output not found for this expert_code"


class SAM2YoloLearnerRunRequest(BaseModel):
    learner_video_id: UUID | None = None
    learner_video_path: str | None = None
    attempt_id: UUID | None = None
    compare_session_id: str | None = None
    chapter_id: UUID | None = None
    expert_code: str | None = None
    stride: int = Field(default=DEFAULT_FRAME_STRIDE, ge=1)
    max_processed_frames: int | None = Field(default=None, ge=0)
    tracking_point_type: str = TRACKING_POINT_TYPE
    save_debug: bool = False


@router.post("/run-learner")
async def run_learner_sam2_yolo(
    request: Request,
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    body, uploaded_video_path = await _parse_run_learner_request(request)
    video_path = uploaded_video_path or _resolve_learner_video_path(body=body, db=db)

    try:
        summary = run_sam2_yolo_scissors_tracking(
            video_path=video_path,
            output_root=SAM2_YOLO_RUNS_ROOT,
            frame_stride=body.stride or DEFAULT_FRAME_STRIDE,
            tracking_point_type=body.tracking_point_type,
            use_gpu=True,
            save_debug=body.save_debug,
            max_processed_frames=body.max_processed_frames,
        )
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001 - surface model/runtime failures cleanly
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"SAM2+YOLO learner processing failed: {exc}",
        ) from exc

    return _build_learner_response(summary=summary, expert_code=body.expert_code)


@router.get("/runs/{run_id}")
def get_sam2_yolo_run(run_id: str) -> dict[str, Any]:
    run_dir = SAM2_YOLO_RUNS_ROOT / _validate_run_id(run_id)
    summary_path = run_dir / SUMMARY_FILENAME
    if not summary_path.is_file():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"SAM2+YOLO run not found: {run_id}",
        )

    summary = _read_json(summary_path)
    return _build_learner_response(summary=summary, expert_code=None)


def _resolve_learner_video_path(*, body: SAM2YoloLearnerRunRequest, db: Session) -> Path:
    provided = [
        body.learner_video_path is not None,
        body.learner_video_id is not None,
        body.attempt_id is not None,
    ]
    if sum(provided) != 1:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Provide exactly one of learner_video_path, learner_video_id, or attempt_id.",
        )

    if body.learner_video_path is not None:
        return _resolve_video_path(body.learner_video_path)

    if body.learner_video_id is not None:
        video = db.query(Video).filter(Video.id == str(body.learner_video_id)).first()
        if video is None:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Learner video not found.")
        return _resolve_video_path(video.file_path)

    assert body.attempt_id is not None
    attempt = (
        db.query(Attempt)
        .options(joinedload(Attempt.learner_video))
        .filter(Attempt.id == str(body.attempt_id))
        .first()
    )
    if attempt is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Attempt not found.")
    if attempt.learner_video is None or not attempt.learner_video.file_path:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Attempt does not have a learner video attached.",
        )
    return _resolve_video_path(attempt.learner_video.file_path)


async def _parse_run_learner_request(request: Request) -> tuple[SAM2YoloLearnerRunRequest, Path | None]:
    content_type = request.headers.get("content-type", "").lower()
    if "multipart/form-data" not in content_type:
        payload = await request.json()
        return SAM2YoloLearnerRunRequest.model_validate(payload), None

    form = await request.form()
    upload = form.get("file")
    if upload is None or not isinstance(upload, StarletteUploadFile):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Multipart request must include a learner video file field named 'file'.",
        )

    try:
        extension = validate_upload_file(upload)
        upload_id = str(uuid.uuid4())
        relative_path = Path("uploads") / "sam2_yolo" / "learner" / upload_id / f"original{extension}"
        await save_upload_file(upload, relative_path)
    except UploadValidationError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to save learner video upload: {exc}",
        ) from exc

    body = SAM2YoloLearnerRunRequest(
        learner_video_path=str(relative_path),
        expert_code=_optional_form_string(form.get("expert_code")),
        stride=_optional_form_int(form.get("stride"), DEFAULT_FRAME_STRIDE),
        max_processed_frames=_optional_form_int(
            form.get("max_processed_frames"),
            None,
        ),
        tracking_point_type=_optional_form_string(form.get("tracking_point_type")) or TRACKING_POINT_TYPE,
        save_debug=_optional_form_bool(form.get("save_debug"), False),
    )
    return body, Path(settings.STORAGE_ROOT) / relative_path


def _optional_form_string(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _optional_form_int(value: Any, default: int | None) -> int | None:
    text = _optional_form_string(value)
    if text is None:
        return default
    try:
        return int(text)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Invalid integer: {text}") from exc


def _optional_form_bool(value: Any, default: bool) -> bool:
    text = _optional_form_string(value)
    if text is None:
        return default
    return text.lower() in {"1", "true", "yes", "on"}


def _resolve_video_path(raw_path: str | Path) -> Path:
    path = Path(str(raw_path).strip().strip('"')).expanduser()
    if path.is_absolute():
        resolved = path.resolve()
    else:
        try:
            resolved = (Path(settings.STORAGE_ROOT) / normalize_storage_key(path)).resolve()
        except ValueError as exc:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid learner video path: {raw_path}",
            ) from exc

    if not resolved.is_file():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Learner video file does not exist: {resolved}",
        )
    return resolved


def _build_learner_response(*, summary: dict[str, Any], expert_code: str | None) -> dict[str, Any]:
    response = dict(summary)
    metrics = _safe_read_json(Path(str(summary.get("metrics_json_path", ""))))
    if metrics:
        response["trajectory_metrics"] = metrics.get("trajectory_metrics")
        response["region_metrics"] = metrics.get("region_metrics")
        response["quality_flags"] = metrics.get("quality_flags")

    response["raw_json_url"] = _storage_url_for(response.get("raw_json_path"))
    response["metrics_json_url"] = _storage_url_for(response.get("metrics_json_path"))
    response["summary_json_url"] = _storage_url_for(response.get("summary_json_path"))
    response["overlay_video_url"] = _storage_url_for(response.get("overlay_video_path"))

    expert_reference = _expert_reference_payload(expert_code)
    response.update(expert_reference)
    return response


def _expert_reference_payload(expert_code: str | None) -> dict[str, Any]:
    if not expert_code:
        return {
            "expert_reference_available": False,
            "expert_code": None,
            "expert_raw_json_path": None,
            "expert_metrics_json_path": None,
        }

    expert_dir = SAM2_YOLO_EXPERTS_ROOT / _validate_run_id(expert_code)
    raw_path = expert_dir / RAW_FILENAME
    metrics_path = expert_dir / METRICS_FILENAME
    if raw_path.is_file() and metrics_path.is_file():
        return {
            "expert_reference_available": True,
            "expert_code": expert_code,
            "expert_raw_json_path": str(raw_path),
            "expert_metrics_json_path": str(metrics_path),
            "expert_raw_json_url": _storage_url_for(raw_path),
            "expert_metrics_json_url": _storage_url_for(metrics_path),
        }

    return {
        "expert_reference_available": False,
        "expert_code": expert_code,
        "expert_raw_json_path": None,
        "expert_metrics_json_path": None,
        "expert_warning": EXPERT_MISSING_WARNING,
    }


def _storage_url_for(path_value: str | Path | None) -> str | None:
    if not path_value:
        return None
    path = Path(str(path_value))
    try:
        return build_storage_url(path)
    except ValueError:
        return None


def _safe_read_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        return _read_json(path)
    except Exception:  # noqa: BLE001 - response can still use summary paths
        return None


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _validate_run_id(value: str) -> str:
    cleaned = value.strip()
    if not cleaned:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="run_id cannot be empty.")
    if any(char in cleaned for char in ('/', "\\", ":", "*", "?", '"', "<", ">", "|")):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid run identifier.")
    return cleaned
