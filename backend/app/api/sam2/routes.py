"""FastAPI routes for the learner SAM 2 pipeline.

These endpoints are the Compare Studio counterpart to the existing
``/api/mediapipe`` routes. SAM 2 initialization is now MANUAL: the
learner must click on the object in the video before submitting the
request. The clicked pixel coordinate is the source of truth for the
SAM 2 prompt — MediaPipe is no longer used to derive the init point.

Pipeline (intentionally sequential, not parallel):

    1. Persist the uploaded video under ``storage/sam2/sources/<run_id>/``.
    2. Build the SAM 2 prompt from the manually clicked (init_x, init_y)
       pixel coordinate + a local bounding box around it.
    3. Run SAM 2 with that prompt on the same learner video.
    4. Write ``raw.json`` / ``summary.json`` alongside the rich
       debug artifacts.

Returned payload is explicitly shaped for the Compare Studio right
panel: it always includes ``device``, ``frame_stride`` and (when
available) the ``initialization_debug_image`` path + public URL.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional
from uuid import uuid4

from fastapi import APIRouter, File, Form, HTTPException, UploadFile, status
from pydantic import BaseModel, Field

from app.core.config import settings
from app.core.sam2_constants import SAM2_DEFAULT_FRAME_STRIDE
from app.schemas.sam2.sam2_contract_schema import (
    SAM2RawDocument,
    SAM2SummaryDocument,
)
from app.schemas.sam2.sam2_schema import SAM2RunMeta
from app.services.media_service import build_storage_url
from app.services.sam2 import (
    SAM2AssetsMissingError,
    SAM2ContractArtifacts,
    SAM2DependencyError,
    SAM2Error,
    SAM2GPUOutOfMemoryError,
    SAM2InitError,
    SAM2PipelineError,
    load_sam2_contract_artifacts,
    resolve_pipeline_run_dir,
    run_sam2_from_manual_prompt,
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
    """Minimal MediaPipe summary echoed back with the SAM 2 payload (legacy field)."""

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

    mediapipe: Optional[SAM2LearnerMediaPipeInfo] = None
    warnings: List[str] = Field(default_factory=list)


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


def _load_raw_document(path: Path) -> Optional[SAM2RawDocument]:
    if not path.is_file():
        return None
    try:
        with path.open("r", encoding="utf-8") as handle:
            return SAM2RawDocument.model_validate(json.load(handle))
    except Exception:  # noqa: BLE001 - raw_json_path is already returned
        logger.exception("Failed to re-read raw.json at %s", path)
        return None


def _build_response(sam2: SAM2ContractArtifacts) -> SAM2LearnerResponse:
    raw_document = sam2.raw if sam2.raw is not None else _load_raw_document(sam2.raw_path)

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
        device=sam2.metadata.device,
        frame_stride=sam2.metadata.frame_stride,
        metadata=sam2.metadata,
        summary=sam2.summary,
        raw_preview=raw_document,
        mediapipe=None,
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




# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.post("/process-upload", response_model=SAM2LearnerResponse)
async def process_uploaded_learner_video(
    file: UploadFile = File(...),
    init_x: float = Form(..., description="Clicked X coordinate in video pixel space."),
    init_y: float = Form(..., description="Clicked Y coordinate in video pixel space."),
    box_half_size: int = Form(
        default=40,
        ge=10,
        le=200,
        description=(
            "Half-side of the local constraint box centred on the click (pixels). "
            "Default 40 → 80×80 px total. Keep small to prevent SAM2 from "
            "selecting large background regions."
        ),
    ),
    run_id: Optional[str] = Form(default=None),
    frame_stride: int = Form(default=SAM2_DEFAULT_FRAME_STRIDE, ge=1),
    render_annotation: bool = Form(default=True),
) -> SAM2LearnerResponse:
    """Accept a learner video upload and run SAM 2 from a manually selected point.

    Compare Studio's "Run SAM 2" button posts the practice video along
    with the pixel coordinate (init_x, init_y) the learner clicked on
    the video frame. SAM 2 is initialized from that point — MediaPipe
    is NOT used to derive the prompt here.

    The frontend must collect the click before enabling this endpoint.
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

    try:
        sam2_result = run_sam2_from_manual_prompt(
            source_path,
            init_x=init_x,
            init_y=init_y,
            box_half_size=box_half_size,
            run_id=effective_run_id,
            frame_stride=frame_stride,
            render_annotation=render_annotation,
        )
    except SAM2Error as exc:
        logger.warning("Learner SAM 2 failed (%s): %s", type(exc).__name__, exc)
        _raise_sam2_error(exc)
    except Exception as exc:  # noqa: BLE001 - surface a controlled 500
        logger.exception("Unexpected learner SAM 2 pipeline error.")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"SAM 2 pipeline failed: {exc}",
        ) from exc

    return _build_response(sam2_result)


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
