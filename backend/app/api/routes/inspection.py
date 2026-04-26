"""Temporary unified "model inspection" API.

This endpoint exists for development / debugging only. It lets the
temporary ``/inspect`` page in the frontend load MediaPipe AND SAM 2
results through a single shape, so we don't grow a separate messy page
per model. A placeholder "optical_flow" model is enumerated but not
implemented yet.

Response shape is intentionally loose (``Any`` for the JSON blobs) so
the inspection UI can just dump them into a pretty-printed view. This
is NOT the compare/evaluation contract — the real pipeline consumes
typed schemas from the model-specific services.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, List, Literal, Optional
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.config import settings
from app.core.database import get_db
from app.core.sam2_constants import (
    SAM2_ANNOTATED_FILENAME as SAM2_ANNOTATED,
    SAM2_METADATA_FILENAME as SAM2_METADATA,
    SAM2_RAW_FILENAME as SAM2_RAW,
    SAM2_SUMMARY_FILENAME as SAM2_SUMMARY,
)
from app.models.video import Video
from app.services.media_service import build_storage_url
from app.services.mediapipe.run_service import (
    ANNOTATED_VIDEO_FILENAME as MP_ANNOTATED,
    DETECTIONS_FILENAME as MP_DETECTIONS,
    FEATURES_FILENAME as MP_FEATURES,
    METADATA_FILENAME as MP_METADATA,
    MediaPipeRunError,
    load_run_artifacts as load_mediapipe_run_artifacts,
)
from app.services.sam2.sam2_service import SAM2AssetsMissingError, validate_sam2_assets


logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/inspection", tags=["Inspection (temporary)"])


SUPPORTED_MODELS = ("mediapipe", "sam2", "optical_flow")
ModelLiteral = Literal["mediapipe", "sam2", "optical_flow"]


# ---------------------------------------------------------------------------
# Response schema (shared across models)
# ---------------------------------------------------------------------------

class InspectionModelOption(BaseModel):
    """One row of the dropdown the frontend renders."""

    id: str
    label: str
    implemented: bool = Field(
        default=True,
        description="False for models that are reserved but not yet implemented.",
    )


class InspectionOptionsResponse(BaseModel):
    models: List[InspectionModelOption]


class InspectionPayload(BaseModel):
    """Unified inspection payload. Unused fields are ``None``."""

    model: ModelLiteral
    source: Literal["expert", "run"] = Field(
        ..., description="Which stable / runtime folder this payload was loaded from."
    )
    identifier: str = Field(
        ..., description="Either the expert video id (expert source) or run id."
    )

    annotated_video_url: Optional[str] = None
    annotated_video_path: Optional[str] = None

    metadata_json: Optional[Any] = None
    raw_json: Optional[Any] = Field(
        default=None,
        description=(
            "Per-frame SAM 2 contract document for ``sam2``; "
            "MediaPipe detections.json for ``mediapipe``."
        ),
    )
    summary_json: Optional[Any] = Field(
        default=None,
        description=(
            "Aggregated SAM 2 contract document for ``sam2``; "
            "MediaPipe features.json for ``mediapipe``."
        ),
    )

    warnings: List[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _safe_read_json(path: Path) -> Optional[Any]:
    try:
        return _read_json(path)
    except FileNotFoundError:
        return None
    except Exception as exc:  # noqa: BLE001 - surface as warning, don't 500
        logger.warning("Failed to read %s: %s", path, exc)
        return None


def _storage_relative_key(path: Path) -> Optional[str]:
    try:
        relative = path.resolve().relative_to(Path(settings.STORAGE_ROOT).resolve())
    except ValueError:
        return None
    return relative.as_posix()


def _annotated_url(path: Optional[Path]) -> Optional[str]:
    if path is None or not path.is_file():
        return None
    key = _storage_relative_key(path)
    return build_storage_url(key) if key else None


def _require_expert(db: Session, expert_id: str) -> Video:
    try:
        parsed = UUID(expert_id)
    except (TypeError, ValueError) as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid expert id: {expert_id!r}",
        ) from exc

    video = db.execute(
        select(Video).where(Video.id == str(parsed))
    ).scalar_one_or_none()
    if video is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Expert video not found: {expert_id}",
        )
    if video.video_role != "expert":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Video {expert_id} is not an expert reference.",
        )
    return video


# ---------------------------------------------------------------------------
# Dropdown options endpoint
# ---------------------------------------------------------------------------

@router.get("/models", response_model=InspectionOptionsResponse)
def list_models() -> InspectionOptionsResponse:
    """Return the dropdown options the inspection page renders.

    The frontend is the single source of truth for label rendering, but
    we serve the list here so new models (e.g. optical_flow) can be
    promoted to ``implemented=true`` server-side without a frontend
    release.
    """
    return InspectionOptionsResponse(
        models=[
            InspectionModelOption(id="mediapipe", label="MediaPipe Hands", implemented=True),
            InspectionModelOption(id="sam2", label="SAM 2 (learner local ROI)", implemented=True),
            InspectionModelOption(
                id="optical_flow",
                label="Optical Flow (coming soon)",
                implemented=False,
            ),
        ]
    )


# ---------------------------------------------------------------------------
# MediaPipe inspection
# ---------------------------------------------------------------------------

def _inspect_mediapipe_expert(db: Session, expert_id: str) -> InspectionPayload:
    video = _require_expert(db, expert_id)
    if not (
        video.mediapipe_metadata_path
        and video.mediapipe_detections_path
        and video.mediapipe_features_path
    ):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=(
                "MediaPipe expert reference is not available for this video. "
                "Run process_expert_mediapipe first."
            ),
        )

    storage_root = Path(settings.STORAGE_ROOT)
    metadata_path = storage_root / video.mediapipe_metadata_path
    detections_path = storage_root / video.mediapipe_detections_path
    features_path = storage_root / video.mediapipe_features_path
    annotated_path: Optional[Path] = (
        storage_root / video.mediapipe_annotated_path
        if video.mediapipe_annotated_path
        else None
    )

    warnings: list[str] = []
    if annotated_path is not None and not annotated_path.is_file():
        warnings.append("annotated_video_missing_on_disk")
        annotated_path = None

    return InspectionPayload(
        model="mediapipe",
        source="expert",
        identifier=str(video.id),
        annotated_video_url=_annotated_url(annotated_path),
        annotated_video_path=str(annotated_path) if annotated_path else None,
        metadata_json=_safe_read_json(metadata_path),
        raw_json=_safe_read_json(detections_path),
        summary_json=_safe_read_json(features_path),
        warnings=warnings,
    )


def _inspect_mediapipe_run(run_id: str) -> InspectionPayload:
    try:
        artifacts = load_mediapipe_run_artifacts(run_id)
    except MediaPipeRunError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)
        ) from exc

    return InspectionPayload(
        model="mediapipe",
        source="run",
        identifier=run_id,
        annotated_video_url=_annotated_url(artifacts.annotated_video_path),
        annotated_video_path=str(artifacts.annotated_video_path),
        metadata_json=_safe_read_json(artifacts.metadata_path),
        raw_json=_safe_read_json(artifacts.detections_path),
        summary_json=_safe_read_json(artifacts.features_path),
        warnings=list(artifacts.partial_errors),
    )


# ---------------------------------------------------------------------------
# SAM 2 inspection
# ---------------------------------------------------------------------------

def _inspect_sam2_expert(db: Session, expert_id: str) -> InspectionPayload:
    video = _require_expert(db, expert_id)
    if not (
        video.sam2_metadata_path
        and video.sam2_raw_path
        and video.sam2_summary_path
    ):
        try:
            validate_sam2_assets()
        except SAM2AssetsMissingError as exc:
            debug_payload = exc.to_debug_payload()
            debug_payload["frontend_message"] = (
                "SAM2 expert data cannot be generated because required SAM2 model assets are missing."
            )
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail={
                    "code": "sam2_assets_missing",
                    "error_type": "SAM2AssetsMissingError",
                    "message": "SAM2 assets are missing. Required checkpoint/config files were not found.",
                    "resolved_checkpoint_path": debug_payload["resolved_checkpoint_path"],
                    "resolved_config_path": debug_payload["resolved_config_path"],
                    "checkpoint_exists": debug_payload["checkpoint_exists"],
                    "config_exists": debug_payload["config_exists"],
                    "debug": debug_payload,
                },
            ) from exc
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={
                "code": "sam2_expert_data_missing",
                "message": "SAM2 expert data not found. Please run SAM2 expert preprocessing first.",
            },
        )

    storage_root = Path(settings.STORAGE_ROOT)
    metadata_path = storage_root / video.sam2_metadata_path
    raw_path = storage_root / video.sam2_raw_path
    summary_path = storage_root / video.sam2_summary_path
    annotated_path: Optional[Path] = (
        storage_root / video.sam2_annotated_path
        if video.sam2_annotated_path
        else None
    )

    warnings: list[str] = []
    if annotated_path is not None and not annotated_path.is_file():
        warnings.append("annotated_video_missing_on_disk")
        annotated_path = None

    return InspectionPayload(
        model="sam2",
        source="expert",
        identifier=str(video.id),
        annotated_video_url=_annotated_url(annotated_path),
        annotated_video_path=str(annotated_path) if annotated_path else None,
        metadata_json=_safe_read_json(metadata_path),
        raw_json=_safe_read_json(raw_path),
        summary_json=_safe_read_json(summary_path),
        warnings=warnings,
    )


def _inspect_sam2_run(run_id: str) -> InspectionPayload:
    from app.services.sam2.pipeline_service import (  # local import avoids cold-starting torch
        SAM2PipelineError,
        resolve_run_dir,
    )

    try:
        run_dir = resolve_run_dir(run_id)
    except SAM2PipelineError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)
        ) from exc

    if not run_dir.is_dir():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"SAM 2 run not found: {run_id}",
        )

    metadata_path = run_dir / SAM2_METADATA
    raw_path = run_dir / SAM2_RAW
    summary_path = run_dir / SAM2_SUMMARY
    annotated_path = run_dir / SAM2_ANNOTATED

    warnings: list[str] = []
    for label, path in (
        ("missing_metadata", metadata_path),
        ("missing_raw", raw_path),
        ("missing_summary", summary_path),
    ):
        if not path.is_file():
            warnings.append(label)

    return InspectionPayload(
        model="sam2",
        source="run",
        identifier=run_id,
        annotated_video_url=_annotated_url(annotated_path) if annotated_path.is_file() else None,
        annotated_video_path=str(annotated_path) if annotated_path.is_file() else None,
        metadata_json=_safe_read_json(metadata_path),
        raw_json=_safe_read_json(raw_path),
        summary_json=_safe_read_json(summary_path),
        warnings=warnings,
    )


# ---------------------------------------------------------------------------
# Unified dispatch endpoints
# ---------------------------------------------------------------------------

def _reject_optical_flow() -> None:
    raise HTTPException(
        status_code=status.HTTP_501_NOT_IMPLEMENTED,
        detail=(
            "The optical_flow model is reserved for a future pipeline step "
            "and is not available yet."
        ),
    )


@router.get("/experts/{expert_id}", response_model=InspectionPayload)
def inspect_expert(
    expert_id: str,
    model: ModelLiteral = "mediapipe",
    db: Session = Depends(get_db),
) -> InspectionPayload:
    """Return the inspection payload for a saved expert reference."""
    if model == "mediapipe":
        return _inspect_mediapipe_expert(db, expert_id)
    if model == "sam2":
        return _inspect_sam2_expert(db, expert_id)
    _reject_optical_flow()
    raise HTTPException(status_code=500, detail="unreachable")  # pragma: no cover


@router.get("/runs/{run_id}", response_model=InspectionPayload)
def inspect_run(
    run_id: str,
    model: ModelLiteral = "mediapipe",
) -> InspectionPayload:
    """Return the inspection payload for a transient pipeline run folder."""
    if model == "mediapipe":
        return _inspect_mediapipe_run(run_id)
    if model == "sam2":
        return _inspect_sam2_run(run_id)
    _reject_optical_flow()
    raise HTTPException(status_code=500, detail="unreachable")  # pragma: no cover


# ---------------------------------------------------------------------------
# Expert discovery (used by the inspection page to populate its "pick expert" list)
# ---------------------------------------------------------------------------

class ExpertInspectionSummary(BaseModel):
    expert_video_id: UUID
    chapter_id: Optional[UUID] = None
    title: Optional[str] = None
    file_name: Optional[str] = None
    mediapipe_available: bool = False
    sam2_available: bool = False


@router.get("/experts", response_model=List[ExpertInspectionSummary])
def list_experts(db: Session = Depends(get_db)) -> List[ExpertInspectionSummary]:
    """List experts along with which model references are already saved."""
    from app.models.chapter import Chapter  # local import to avoid cycle

    stmt = (
        select(Video, Chapter)
        .join(Chapter, Video.chapter_id == Chapter.id, isouter=True)
        .where(Video.video_role == "expert")
        .order_by(Video.created_at.desc())
    )

    rows = db.execute(stmt).all()
    summaries: List[ExpertInspectionSummary] = []
    for video, chapter in rows:
        summaries.append(
            ExpertInspectionSummary(
                expert_video_id=UUID(str(video.id)),
                chapter_id=UUID(str(video.chapter_id)) if video.chapter_id else None,
                title=getattr(chapter, "title", None) if chapter is not None else None,
                file_name=video.file_name,
                mediapipe_available=bool(
                    video.mediapipe_status == "completed"
                    and video.mediapipe_metadata_path
                ),
                sam2_available=bool(
                    video.sam2_status == "completed" and video.sam2_metadata_path
                ),
            )
        )
    return summaries
