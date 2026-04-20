"""FastAPI routes for the MediaPipe Hands pipeline.

Exposed endpoints:

    POST /api/mediapipe/process          -> run the full pipeline on a video
    GET  /api/mediapipe/result/{run_id}  -> retrieve artifacts for a prior run

The response payload intentionally returns both a filesystem
``annotated_video_path`` (useful from the backend host) and an
``annotated_video_url`` that is mounted via the existing ``/storage``
static files handler, so the Compare Studio frontend can reference the
annotated video directly.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional
from uuid import uuid4

from fastapi import APIRouter, File, Form, HTTPException, UploadFile, status
from pydantic import BaseModel, Field

from app.core.config import settings
from app.schemas.mediapipe.mediapipe_schema import MediaPipeRunMeta
from app.services.media_service import build_storage_url
from app.services.mediapipe.run_service import (
    MediaPipeRunArtifacts,
    MediaPipeRunError,
    load_run_artifacts,
    run_pipeline,
)
from app.services.upload_service import (
    UploadValidationError,
    save_upload_file,
    validate_upload_file,
)


logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/mediapipe", tags=["MediaPipe"])


# ---------------------------------------------------------------------------
# Request / response schemas
# ---------------------------------------------------------------------------

class MediaPipeProcessRequest(BaseModel):
    """Body accepted by ``POST /api/mediapipe/process``."""

    video_path: str = Field(
        ...,
        description=(
            "Path to the learner (or expert) video. Accepts an absolute "
            "path, a path relative to the backend working directory, a "
            "path relative to the storage root (e.g. "
            "'expert/<chapter_id>/<file>.mp4'), or a public URL that "
            "starts with '/storage/'."
        ),
    )
    run_id: Optional[str] = Field(
        default=None,
        description="Optional deterministic run id. When omitted a UUID4 hex is generated.",
    )
    max_num_hands: int = Field(default=2, ge=1, le=4)
    min_detection_confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    min_tracking_confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    render_annotation: bool = Field(
        default=True,
        description="When false, skip writing annotated.mp4 (JSON artifacts are always written).",
    )


class MediaPipeRunSummary(BaseModel):
    """Subset of ``metadata.json`` returned on API responses."""

    run_id: str
    source_video_path: str
    fps: float
    frame_count: int
    width: int
    height: int
    created_at: str
    selected_hand_policy: str
    total_frames: int
    frames_with_detection: int
    detection_rate: float
    right_hand_selected_count: int
    left_hand_selected_count: int


class MediaPipeRunPaths(BaseModel):
    """Filesystem + URL pointers to the artifacts of a run."""

    run_id: str
    run_folder: str
    detections_json_path: str
    features_json_path: str
    metadata_json_path: str
    annotated_video_path: str
    annotated_video_url: Optional[str] = Field(
        default=None,
        description=(
            "Public URL under the /storage mount. Null if the annotated "
            "video lives outside the storage root for some reason."
        ),
    )


class MediaPipeProcessResponse(BaseModel):
    """Payload returned by both process and result endpoints."""

    run_id: str
    run_folder: str
    detections_json_path: str
    features_json_path: str
    metadata_json_path: str
    annotated_video_path: str
    annotated_video_url: Optional[str] = None
    summary: MediaPipeRunSummary
    partial_errors: List[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _summary_from_metadata(metadata: MediaPipeRunMeta) -> MediaPipeRunSummary:
    return MediaPipeRunSummary(
        run_id=metadata.run_id,
        source_video_path=metadata.source_video_path,
        fps=metadata.fps,
        frame_count=metadata.frame_count,
        width=metadata.width,
        height=metadata.height,
        created_at=metadata.created_at,
        selected_hand_policy=metadata.selected_hand_policy,
        total_frames=metadata.total_frames,
        frames_with_detection=metadata.frames_with_detection,
        detection_rate=metadata.detection_rate,
        right_hand_selected_count=metadata.right_hand_selected_count,
        left_hand_selected_count=metadata.left_hand_selected_count,
    )


def _storage_relative_key(path: Path) -> Optional[str]:
    """Return ``path`` expressed relative to ``settings.STORAGE_ROOT`` (POSIX)."""
    try:
        relative = path.resolve().relative_to(Path(settings.STORAGE_ROOT).resolve())
    except ValueError:
        return None
    return relative.as_posix()


def _build_response(artifacts: MediaPipeRunArtifacts) -> MediaPipeProcessResponse:
    annotated_storage_key = _storage_relative_key(artifacts.annotated_video_path)
    annotated_url = (
        build_storage_url(annotated_storage_key) if annotated_storage_key else None
    )

    return MediaPipeProcessResponse(
        run_id=artifacts.run_id,
        run_folder=str(artifacts.run_dir),
        detections_json_path=str(artifacts.detections_path),
        features_json_path=str(artifacts.features_path),
        metadata_json_path=str(artifacts.metadata_path),
        annotated_video_path=str(artifacts.annotated_video_path),
        annotated_video_url=annotated_url,
        summary=_summary_from_metadata(artifacts.metadata),
        partial_errors=list(artifacts.partial_errors),
    )


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.post("/process", response_model=MediaPipeProcessResponse)
def process_video(payload: MediaPipeProcessRequest) -> MediaPipeProcessResponse:
    """Run the full MediaPipe pipeline on a single learner/expert video.

    Produces ``detections.json``, ``features.json``, ``metadata.json`` and
    ``annotated.mp4`` inside a fresh run folder under
    ``backend/storage/mediapipe/runs/<run_id>/``.
    """
    try:
        artifacts = run_pipeline(
            payload.video_path,
            run_id=payload.run_id,
            max_num_hands=payload.max_num_hands,
            min_detection_confidence=payload.min_detection_confidence,
            min_tracking_confidence=payload.min_tracking_confidence,
            render_annotation=payload.render_annotation,
        )
    except MediaPipeRunError as exc:
        logger.warning("MediaPipe process failed: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc
    except Exception as exc:  # noqa: BLE001 - we want a controlled 500
        logger.exception("Unexpected MediaPipe process error.")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"MediaPipe pipeline failed: {exc}",
        ) from exc

    return _build_response(artifacts)


@router.get("/result/{run_id}", response_model=MediaPipeProcessResponse)
def get_run_result(run_id: str) -> MediaPipeProcessResponse:
    """Return artifact paths and summary for a previously completed run."""
    try:
        artifacts = load_run_artifacts(run_id)
    except MediaPipeRunError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(exc),
        ) from exc

    return _build_response(artifacts)


@router.post("/process-upload", response_model=MediaPipeProcessResponse)
async def process_uploaded_video(
    file: UploadFile = File(...),
    run_id: Optional[str] = Form(default=None),
    max_num_hands: int = Form(default=2),
    min_detection_confidence: float = Form(default=0.5),
    min_tracking_confidence: float = Form(default=0.5),
    render_annotation: bool = Form(default=True),
) -> MediaPipeProcessResponse:
    """Accept a multipart learner video upload and run the full pipeline.

    This exists so the Compare Studio frontend can POST the learner's
    freshly-uploaded blob directly, without a separate upload step. The
    raw file is persisted under
    ``backend/storage/mediapipe/sources/<run_id>/<filename>`` so the
    annotated video and JSON documents can reference a stable filesystem
    path later.
    """
    try:
        extension = validate_upload_file(file)
    except UploadValidationError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)
        ) from exc

    effective_run_id = run_id or uuid4().hex
    source_relative_path = (
        Path("mediapipe") / "sources" / effective_run_id / f"source{extension}"
    )

    try:
        await save_upload_file(file, source_relative_path)
    except UploadValidationError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)
        ) from exc
    except Exception as exc:  # noqa: BLE001 - surface a controlled 500
        logger.exception("Failed to persist uploaded MediaPipe source video.")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to store uploaded video.",
        ) from exc

    source_path = Path(settings.STORAGE_ROOT) / source_relative_path

    try:
        artifacts = run_pipeline(
            source_path,
            run_id=effective_run_id,
            max_num_hands=max_num_hands,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
            render_annotation=render_annotation,
        )
    except MediaPipeRunError as exc:
        logger.warning("MediaPipe process-upload failed: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)
        ) from exc
    except Exception as exc:  # noqa: BLE001
        logger.exception("Unexpected MediaPipe process-upload error.")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"MediaPipe pipeline failed: {exc}",
        ) from exc

    return _build_response(artifacts)
