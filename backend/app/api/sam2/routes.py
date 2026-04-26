"""FastAPI routes for the learner SAM 2 pipeline.

These endpoints are the Compare Studio counterpart to the existing
``/api/mediapipe`` routes: they accept a learner/practice video, run
the full MediaPipe -> SAM 2 pipeline, and return the stable contract
artifacts (raw/summary/metadata + annotated video) plus the
initialization debug image used to build the SAM 2 prompt.

Pipeline (intentionally sequential, not parallel):

    1. Persist the uploaded video under ``storage/sam2/sources/<run_id>/``.
    2. Run MediaPipe Hands on the learner video — its ``features.json``
       is the source of truth for the SAM 2 auto-prompt.
    3. Build the SAM 2 prompt from MediaPipe (point + padded hand bbox,
       averaged over the first few valid frames).
    4. Run SAM 2 with that prompt on the same learner video.
    5. Write ``raw.json`` / ``summary.json`` alongside the rich
       debug artifacts.

Returned payload is explicitly shaped for the Compare Studio right
panel: it always includes ``device``, ``frame_stride`` and (when
available) the ``initialization_debug_image`` path + public URL.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional
from uuid import uuid4

from fastapi import APIRouter, File, Form, HTTPException, UploadFile, status
from pydantic import BaseModel, Field

from app.core.config import settings
from app.core.sam2_constants import (
    SAM2_DEFAULT_FRAME_STRIDE,
    SAM2_MANUAL_INIT_PROMPT_FILENAME,
    SAM2_TIP_ANNOTATED_FILENAME,
    SAM2_TIP_INIT_FILENAME,
    SAM2_TIP_TRACKING_FILENAME,
)
from app.schemas.sam2.sam2_contract_schema import (
    SAM2RawDocument,
    SAM2SummaryDocument,
)
from app.schemas.sam2.sam2_schema import SAM2RunMeta
from app.schemas.sam2.tip_tracking_schema import SAM2TipTrackingDocument
from app.services.media_service import build_storage_url
from app.services.mediapipe.run_service import (
    MediaPipeRunArtifacts,
    MediaPipeRunError,
    run_pipeline as run_mediapipe_pipeline,
)
from app.services.sam2 import (
    SAM2AssetsMissingError,
    SAM2ContractArtifacts,
    SAM2DependencyError,
    SAM2Error,
    SAM2GPUOutOfMemoryError,
    SAM2InitError,
    SAM2PipelineError,
    SAM2TipTrackingError,
    build_manual_init_prompt,
    load_sam2_contract_artifacts,
    resolve_pipeline_run_dir,
    run_sam2_pipeline,
    run_sam2_from_mediapipe_prompt,
    track_scissor_tip_in_sam_roi,
)
from app.services.upload_service import (
    UploadValidationError,
    save_upload_file,
    validate_upload_file,
)


logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/sam2", tags=["SAM2"])


# ---------------------------------------------------------------------------
# Response schemas
# ---------------------------------------------------------------------------

class SAM2LearnerMediaPipeInfo(BaseModel):
    """Minimal MediaPipe summary echoed back with the SAM 2 payload."""

    run_id: str
    run_folder: str
    features_json_path: str
    metadata_json_path: str
    annotated_video_path: str
    annotated_video_url: Optional[str] = None
    total_frames: int
    frames_with_detection: int
    detection_rate: float


class SAM2LearnerResponse(BaseModel):
    """Payload returned by the learner SAM 2 endpoints."""

    run_id: str
    run_folder: str

    raw_json_path: str
    summary_json_path: str
    metadata_json_path: str
    annotated_video_path: Optional[str] = None
    annotated_video_url: Optional[str] = None
    source_video_path: str

    initialization_debug_image_path: Optional[str] = None
    initialization_debug_image_url: Optional[str] = None
    tip_initialization_json_path: Optional[str] = None
    tip_initialization_json_url: Optional[str] = None
    tip_tracking_json_path: Optional[str] = None
    tip_tracking_json_url: Optional[str] = None
    tip_annotated_video_path: Optional[str] = None
    tip_annotated_video_url: Optional[str] = None
    manual_init_prompt_path: Optional[str] = None
    manual_init_prompt_url: Optional[str] = None

    device: str = Field(..., description="Compute device used by SAM 2 ('cuda' or 'cpu').")
    frame_stride: int = Field(..., description="Frame stride SAM 2 actually processed with.")

    metadata: SAM2RunMeta
    summary: SAM2SummaryDocument
    raw_preview: Optional[SAM2RawDocument] = Field(
        default=None,
        description=(
            "Full raw.json document for this run. Included eagerly for now; "
            "if this ever becomes too large we can swap it for a truncated "
            "preview without changing the response shape."
        ),
    )
    tip_tracking_preview: Optional[SAM2TipTrackingDocument] = None
    manual_init_prompt_preview: Optional[Dict[str, Any]] = None

    mediapipe: Optional[SAM2LearnerMediaPipeInfo] = None
    warnings: List[str] = Field(default_factory=list)


class SAM2TipInitializationRequest(BaseModel):
    frame_index: int = Field(default=0, ge=0, description="Processed-frame index seed (0-based).")
    tip_point: List[float] = Field(..., min_length=2, max_length=2)


class SAM2ManualInitPromptPayload(BaseModel):
    frame_index: int = Field(default=0, ge=0)
    init_mode: str = Field(default="manual")
    box: Optional[List[float]] = Field(default=None, min_length=4, max_length=4)
    positive_points: List[List[float]] = Field(default_factory=list)
    negative_points: List[List[float]] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _storage_relative_key(path: Path) -> Optional[str]:
    """Return ``path`` expressed relative to ``settings.STORAGE_ROOT`` (POSIX)."""
    try:
        relative = path.resolve().relative_to(Path(settings.STORAGE_ROOT).resolve())
    except ValueError:
        return None
    return relative.as_posix()


def _storage_url_for(path: Optional[Path]) -> Optional[str]:
    if path is None:
        return None
    key = _storage_relative_key(path)
    return build_storage_url(key) if key else None


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def _mediapipe_info(artifacts: MediaPipeRunArtifacts) -> SAM2LearnerMediaPipeInfo:
    return SAM2LearnerMediaPipeInfo(
        run_id=artifacts.run_id,
        run_folder=str(artifacts.run_dir),
        features_json_path=str(artifacts.features_path),
        metadata_json_path=str(artifacts.metadata_path),
        annotated_video_path=str(artifacts.annotated_video_path),
        annotated_video_url=_storage_url_for(artifacts.annotated_video_path),
        total_frames=artifacts.metadata.total_frames,
        frames_with_detection=artifacts.metadata.frames_with_detection,
        detection_rate=artifacts.metadata.detection_rate,
    )


def _load_raw_document(path: Path) -> Optional[SAM2RawDocument]:
    if not path.is_file():
        return None
    try:
        with path.open("r", encoding="utf-8") as handle:
            return SAM2RawDocument.model_validate(json.load(handle))
    except Exception:  # noqa: BLE001 - raw_json_path is already returned
        logger.exception("Failed to re-read raw.json at %s", path)
        return None


def _load_tip_tracking_document(path: Path) -> Optional[SAM2TipTrackingDocument]:
    if not path.is_file():
        return None
    try:
        with path.open("r", encoding="utf-8") as handle:
            return SAM2TipTrackingDocument.model_validate(json.load(handle))
    except Exception:  # noqa: BLE001
        logger.exception("Failed to re-read tip tracking json at %s", path)
        return None


def _load_manual_prompt(path: Path) -> Optional[Dict[str, Any]]:
    if not path.is_file():
        return None
    try:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if isinstance(payload, dict):
            return payload
        return None
    except Exception:  # noqa: BLE001
        logger.exception("Failed to read manual init prompt at %s", path)
        return None


def _build_response(
    sam2: SAM2ContractArtifacts,
    *,
    mediapipe: Optional[MediaPipeRunArtifacts] = None,
) -> SAM2LearnerResponse:
    raw_document = sam2.raw if sam2.raw is not None else _load_raw_document(sam2.raw_path)
    tip_init_path = sam2.run_dir / SAM2_TIP_INIT_FILENAME
    tip_tracking_path = sam2.run_dir / SAM2_TIP_TRACKING_FILENAME
    tip_annotated_path = sam2.run_dir / SAM2_TIP_ANNOTATED_FILENAME
    manual_prompt_path = sam2.run_dir / SAM2_MANUAL_INIT_PROMPT_FILENAME
    tip_tracking_document = _load_tip_tracking_document(tip_tracking_path)
    manual_prompt_preview = _load_manual_prompt(manual_prompt_path)

    return SAM2LearnerResponse(
        run_id=sam2.run_id,
        run_folder=str(sam2.run_dir),
        raw_json_path=str(sam2.raw_path),
        summary_json_path=str(sam2.summary_path),
        metadata_json_path=str(sam2.metadata_path),
        annotated_video_path=(
            str(sam2.annotated_video_path) if sam2.annotated_video_path else None
        ),
        annotated_video_url=_storage_url_for(sam2.annotated_video_path),
        source_video_path=str(sam2.source_video_path),
        initialization_debug_image_path=(
            str(sam2.init_debug_image_path) if sam2.init_debug_image_path else None
        ),
        initialization_debug_image_url=_storage_url_for(sam2.init_debug_image_path),
        tip_initialization_json_path=str(tip_init_path) if tip_init_path.is_file() else None,
        tip_initialization_json_url=_storage_url_for(tip_init_path) if tip_init_path.is_file() else None,
        tip_tracking_json_path=str(tip_tracking_path) if tip_tracking_path.is_file() else None,
        tip_tracking_json_url=_storage_url_for(tip_tracking_path) if tip_tracking_path.is_file() else None,
        tip_annotated_video_path=str(tip_annotated_path) if tip_annotated_path.is_file() else None,
        tip_annotated_video_url=_storage_url_for(tip_annotated_path) if tip_annotated_path.is_file() else None,
        manual_init_prompt_path=str(manual_prompt_path) if manual_prompt_path.is_file() else None,
        manual_init_prompt_url=_storage_url_for(manual_prompt_path) if manual_prompt_path.is_file() else None,
        device=sam2.metadata.device,
        frame_stride=sam2.metadata.frame_stride,
        metadata=sam2.metadata,
        summary=sam2.summary,
        raw_preview=raw_document,
        tip_tracking_preview=tip_tracking_document,
        manual_init_prompt_preview=manual_prompt_preview,
        mediapipe=_mediapipe_info(mediapipe) if mediapipe is not None else None,
        warnings=list(sam2.warnings),
    )


def _raise_sam2_error(exc: SAM2Error) -> None:
    """Translate domain errors into a structured HTTPException."""
    detail: Dict[str, Any] = {
        "error_type": type(exc).__name__,
        "message": str(exc),
    }

    if isinstance(exc, SAM2AssetsMissingError):
        try:
            detail.update(exc.to_debug_payload() or {})
        except Exception:  # noqa: BLE001
            pass
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=detail,
        ) from exc
    if isinstance(exc, SAM2DependencyError):
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=detail,
        ) from exc
    if isinstance(exc, SAM2GPUOutOfMemoryError):
        detail["resolved_device"] = getattr(exc, "resolved_device", None)
        detail["hint"] = (
            "SAM 2 ran out of GPU memory. Another SAM 2 process is holding "
            "the GPU. Close the other consumer (e.g. a running expert SAM 2 "
            "CLI) and try again."
        )
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=detail,
        ) from exc
    if isinstance(exc, SAM2InitError):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=detail,
        ) from exc
    if isinstance(exc, SAM2PipelineError):
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=detail,
        ) from exc
    raise HTTPException(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        detail=detail,
    ) from exc


@dataclass
class _LearnerPipelineResult:
    mediapipe: MediaPipeRunArtifacts
    sam2: SAM2ContractArtifacts


def _run_learner_pipeline(
    source_path: Path,
    *,
    run_id: str,
    frame_stride: int,
    render_annotation: bool,
    max_num_hands: int,
    min_detection_confidence: float,
    min_tracking_confidence: float,
    manual_init_sam: bool,
    auto_local_roi_sam: bool,
    manual_prompt: Optional[SAM2ManualInitPromptPayload] = None,
) -> _LearnerPipelineResult:
    """Run MediaPipe -> SAM 2 on a learner video.

    Errors from MediaPipe bubble up as ``MediaPipeRunError``; SAM 2
    errors bubble up as ``SAM2Error`` subclasses so the route can
    translate them into structured HTTP responses.
    """
    mediapipe_artifacts = run_mediapipe_pipeline(
        source_path,
        run_id=run_id,
        max_num_hands=max_num_hands,
        min_detection_confidence=min_detection_confidence,
        min_tracking_confidence=min_tracking_confidence,
        render_annotation=render_annotation,
    )

    if manual_init_sam:
        if manual_prompt is None:
            raise SAM2InitError(
                "manual_init_sam=true requires a manual initialization prompt."
            )
        init_prompt = build_manual_init_prompt(
            frame_index=int(manual_prompt.frame_index),
            box_xyxy=manual_prompt.box,
            positive_points=list(manual_prompt.positive_points),
            negative_points=list(manual_prompt.negative_points),
        )
        underlying = run_sam2_pipeline(
            source_path,
            init_prompt,
            run_id=run_id,
            frame_stride=frame_stride,
            render_annotation=render_annotation,
        )
        from app.services.sam2.pipeline_service import _finalize_contract_artifacts

        sam2_artifacts = _finalize_contract_artifacts(underlying)
        prompt_path = sam2_artifacts.run_dir / SAM2_MANUAL_INIT_PROMPT_FILENAME
        _write_json(prompt_path, manual_prompt.model_dump(mode="json"))
    elif auto_local_roi_sam:
        sam2_artifacts = run_sam2_from_mediapipe_prompt(
            source_path,
            mediapipe_artifacts.features_path,
            run_id=run_id,
            frame_stride=frame_stride,
            render_annotation=render_annotation,
        )
    else:
        raise SAM2InitError(
            "Invalid SAM2 mode: both manual_init_sam and auto_local_roi_sam are false."
        )

    return _LearnerPipelineResult(
        mediapipe=mediapipe_artifacts,
        sam2=sam2_artifacts,
    )


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.post("/process-upload", response_model=SAM2LearnerResponse)
async def process_uploaded_learner_video(
    file: UploadFile = File(...),
    run_id: Optional[str] = Form(default=None),
    frame_stride: int = Form(default=SAM2_DEFAULT_FRAME_STRIDE, ge=1),
    render_annotation: bool = Form(default=True),
    manual_init_sam: bool = Form(default=True),
    auto_local_roi_sam: bool = Form(default=False),
    manual_prompt_json: Optional[str] = Form(default=None),
    max_num_hands: int = Form(default=2, ge=1, le=4),
    min_detection_confidence: float = Form(default=0.5, ge=0.0, le=1.0),
    min_tracking_confidence: float = Form(default=0.5, ge=0.0, le=1.0),
) -> SAM2LearnerResponse:
    """Accept a learner video upload and run MediaPipe -> SAM 2 end-to-end.

    Compare Studio's "Run SAM 2" button posts the freshly-uploaded
    practice video here. We always run MediaPipe first because the SAM
    2 initialization prompt is derived from its output — there is no
    manual point picking on the learner side.
    """
    try:
        extension = validate_upload_file(file)
    except UploadValidationError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)
        ) from exc

    effective_run_id = run_id or uuid4().hex
    source_relative_path = (
        Path("sam2") / "sources" / effective_run_id / f"source{extension}"
    )

    try:
        await save_upload_file(file, source_relative_path)
    except UploadValidationError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)
        ) from exc
    except Exception as exc:  # noqa: BLE001 - surface a controlled 500
        logger.exception("Failed to persist uploaded SAM 2 learner video.")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to store uploaded video.",
        ) from exc

    source_path = Path(settings.STORAGE_ROOT) / source_relative_path

    parsed_manual_prompt: Optional[SAM2ManualInitPromptPayload] = None
    if manual_init_sam:
        logger.info(
            "SAM2 manual init form received: manual_init_sam=%s has_manual_prompt_json=%s",
            bool(manual_init_sam),
            bool(manual_prompt_json),
        )
        if not manual_prompt_json:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="manual_init_sam=true requires manual_prompt_json.",
            )
        try:
            payload = json.loads(manual_prompt_json)
            parsed_manual_prompt = SAM2ManualInitPromptPayload.model_validate(payload)
            logger.info(
                "SAM2 manual prompt parsed: frame_index=%s has_box=%s positive_points=%s negative_points=%s payload=%s",
                int(parsed_manual_prompt.frame_index),
                bool(parsed_manual_prompt.box),
                len(parsed_manual_prompt.positive_points),
                len(parsed_manual_prompt.negative_points),
                payload,
            )
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=f"Invalid manual_prompt_json: {exc}",
            ) from exc

    try:
        result = _run_learner_pipeline(
            source_path,
            run_id=effective_run_id,
            frame_stride=frame_stride,
            render_annotation=render_annotation,
            max_num_hands=max_num_hands,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
            manual_init_sam=manual_init_sam,
            auto_local_roi_sam=auto_local_roi_sam,
            manual_prompt=parsed_manual_prompt,
        )
    except MediaPipeRunError as exc:
        logger.warning("Learner SAM 2: MediaPipe prerequisite failed: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "error_type": "MediaPipeRunError",
                "message": str(exc),
                "stage": "mediapipe",
            },
        ) from exc
    except SAM2Error as exc:
        logger.warning("Learner SAM 2 failed (%s): %s", type(exc).__name__, exc)
        _raise_sam2_error(exc)
    except Exception as exc:  # noqa: BLE001 - surface a controlled 500
        logger.exception("Unexpected learner SAM 2 pipeline error.")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"SAM 2 pipeline failed: {exc}",
        ) from exc

    return _build_response(result.sam2, mediapipe=result.mediapipe)


@router.post("/track-tip/{run_id}", response_model=SAM2LearnerResponse)
def track_tip_for_learner_run(
    run_id: str,
    request: SAM2TipInitializationRequest,
) -> SAM2LearnerResponse:
    """Seed and track the learner scissor tip inside the SAM hand ROI."""
    try:
        run_dir = resolve_pipeline_run_dir(run_id)
    except SAM2PipelineError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)
        ) from exc

    if not run_dir.is_dir():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"SAM 2 run '{run_id}' was not found.",
        )

    try:
        artifacts = load_sam2_contract_artifacts(run_dir)
    except SAM2PipelineError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)
        ) from exc

    tip_xy = request.tip_point
    try:
        track_scissor_tip_in_sam_roi(
            run_dir=Path(run_dir),
            raw_document=artifacts.raw,
            frame_index=int(request.frame_index),
            tip_point_xy=[float(tip_xy[0]), float(tip_xy[1])],
        )
    except SAM2TipTrackingError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={
                "error_type": "SAM2TipTrackingError",
                "message": str(exc),
            },
        ) from exc
    except Exception as exc:  # noqa: BLE001
        logger.exception("Unexpected tip tracking failure for run %s", run_id)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error_type": "SAM2TipTrackingError",
                "message": f"Tip tracking failed: {exc}",
            },
        ) from exc

    # Reload so response picks up tip artifacts.
    refreshed = load_sam2_contract_artifacts(run_dir)
    return _build_response(refreshed)


@router.get("/result/{run_id}", response_model=SAM2LearnerResponse)
def get_learner_run_result(run_id: str) -> SAM2LearnerResponse:
    """Return the SAM 2 contract artifacts for a previously completed run."""
    try:
        run_dir = resolve_pipeline_run_dir(run_id)
    except SAM2PipelineError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)
        ) from exc

    if not run_dir.is_dir():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"SAM 2 run '{run_id}' was not found.",
        )

    try:
        artifacts = load_sam2_contract_artifacts(run_dir)
    except SAM2PipelineError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)
        ) from exc

    return _build_response(artifacts)
