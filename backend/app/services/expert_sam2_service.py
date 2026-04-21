"""One-time expert SAM 2 preprocessing service.

Expert videos are processed EXACTLY ONCE via this service (either from
the ``app/scripts/process_expert_sam2.py`` CLI or the admin-only
route). The outputs are saved into a stable, reusable folder:

    backend/storage/expert/sam2/{expert_code}/
        source.mp4
        raw.json
        summary.json
        metadata.json
        annotated.mp4   (optional — best-effort)

``expert_code`` is the expert ``Video.id`` (UUID string) — the same
identifier the MediaPipe expert flow uses. No parallel "expert code"
concept is introduced.

This service:
    * never runs during the learner compare / evaluation flow
    * never creates a parallel expert record in the DB
    * only UPDATES the existing expert ``Video`` row with SAM 2 file
      paths and summary metadata
    * requires the MediaPipe expert reference to exist first: SAM 2 is
      initialized automatically from ``features.json`` (see
      ``build_init_prompt_from_mediapipe``), never from a UI click.

Design is a deliberate mirror of ``expert_mediapipe_service.py`` so the
two flows stay easy to reason about side by side.
"""

from __future__ import annotations

import json
import logging
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from typing import Optional
from uuid import UUID, uuid4

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.config import settings
from app.core.sam2_constants import (
    SAM2_ANNOTATED_FILENAME,
    SAM2_EXPERT_SUBDIR,
    SAM2_INIT_DEBUG_IMAGE_FILENAME,
    SAM2_METADATA_FILENAME,
    SAM2_RAW_FILENAME,
    SAM2_SOURCE_FILENAME,
    SAM2_SUMMARY_FILENAME,
)
from app.models.chapter import Chapter
from app.models.video import Video
from app.schemas.expert_sam2_schema import (
    ExpertSam2Reference,
    ExpertSam2Summary,
)
from app.schemas.sam2.sam2_contract_schema import (
    SAM2SummaryDocument,
)
from app.schemas.sam2.sam2_schema import SAM2RunMeta
from app.services.sam2.pipeline_service import (
    SAM2ContractArtifacts,
    SAM2PipelineError,
    copy_contract_artifacts_to,
    run_sam2_from_mediapipe_prompt,
)
from app.services.sam2.sam2_service import (
    SAM2AssetsMissingError,
    SAM2DependencyError,
    SAM2Error,
    SAM2GPUOutOfMemoryError,
    SAM2InitError,
    resolve_video_path,
)


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SOURCE_FILENAME = SAM2_SOURCE_FILENAME  # "source.mp4"
RAW_FILENAME = SAM2_RAW_FILENAME        # "raw.json"
SUMMARY_FILENAME = SAM2_SUMMARY_FILENAME  # "summary.json"
METADATA_FILENAME = SAM2_METADATA_FILENAME  # "metadata.json"
ANNOTATED_FILENAME = SAM2_ANNOTATED_FILENAME  # "annotated.mp4"
INIT_DEBUG_IMAGE_FILENAME = SAM2_INIT_DEBUG_IMAGE_FILENAME  # "init_prompt_debug.png"

STATUS_PENDING = "pending"
STATUS_PROCESSING = "processing"
STATUS_COMPLETED = "completed"
STATUS_FAILED = "failed"


class ExpertSam2Error(RuntimeError):
    """Raised when the expert SAM 2 preprocessing flow cannot complete."""


class ExpertNotFoundError(ExpertSam2Error):
    """Raised when the requested expert Video row does not exist."""


class ExpertMediaPipeMissingError(ExpertSam2Error):
    """Raised when the expert MediaPipe reference hasn't been computed yet."""


class ExpertSam2AssetsMissingError(ExpertSam2Error):
    """Raised when required SAM 2 model assets are missing on disk."""

    def __init__(self, message: str, *, debug: dict[str, object]):
        super().__init__(message)
        self.debug = debug


class ExpertSam2GPUOutOfMemoryError(ExpertSam2Error):
    """Raised when SAM 2 ran out of GPU memory during an expert run."""

    def __init__(self, message: str, *, resolved_device: Optional[str]):
        super().__init__(message)
        self.resolved_device = resolved_device


@dataclass(frozen=True)
class _ResolvedExpert:
    video: Video
    chapter: Optional[Chapter]


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------

def get_expert_sam2_root() -> Path:
    """Return the stable ``storage/expert/sam2`` folder."""
    return Path(settings.STORAGE_ROOT) / SAM2_EXPERT_SUBDIR


def ensure_expert_sam2_dir(expert_code: str) -> Path:
    """Create (if needed) and return the stable expert folder."""
    if not expert_code or any(token in expert_code for token in ("/", "\\", "..")):
        raise ExpertSam2Error(f"Invalid expert_code: {expert_code!r}")
    folder = get_expert_sam2_root() / expert_code
    folder.mkdir(parents=True, exist_ok=True)
    return folder


def _expert_file_paths(expert_folder: Path) -> dict[str, Path]:
    return {
        "source": expert_folder / SOURCE_FILENAME,
        "raw": expert_folder / RAW_FILENAME,
        "summary": expert_folder / SUMMARY_FILENAME,
        "metadata": expert_folder / METADATA_FILENAME,
        "annotated": expert_folder / ANNOTATED_FILENAME,
        "init_debug_image": expert_folder / INIT_DEBUG_IMAGE_FILENAME,
    }


def _storage_relative(path: Path) -> str:
    """Return ``path`` relative to STORAGE_ROOT as a POSIX string."""
    try:
        return path.resolve().relative_to(Path(settings.STORAGE_ROOT).resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def _resolve_expert_mediapipe_features_path(video: Video) -> Path:
    """Find the MediaPipe features.json that this expert's SAM 2 run should seed from."""
    if not video.mediapipe_features_path:
        raise ExpertMediaPipeMissingError(
            f"Expert {video.id} has no MediaPipe features yet. "
            "Run process_expert_mediapipe first."
        )
    # Stored as a storage-relative POSIX path.
    candidate = Path(settings.STORAGE_ROOT) / video.mediapipe_features_path
    if candidate.is_file():
        return candidate.resolve()

    absolute = Path(video.mediapipe_features_path)
    if absolute.is_absolute() and absolute.is_file():
        return absolute

    raise ExpertMediaPipeMissingError(
        "Expert MediaPipe features.json is missing on disk: "
        f"{video.mediapipe_features_path}"
    )


# ---------------------------------------------------------------------------
# DB resolution
# ---------------------------------------------------------------------------

def _resolve_expert(
    db: Session,
    *,
    expert_video_id: Optional[str | UUID],
    chapter_id: Optional[str | UUID],
) -> _ResolvedExpert:
    if expert_video_id is None and chapter_id is None:
        raise ExpertSam2Error(
            "Either expert_video_id or chapter_id must be provided."
        )

    video: Optional[Video] = None
    if expert_video_id is not None:
        video = db.execute(
            select(Video).where(Video.id == str(expert_video_id))
        ).scalar_one_or_none()
        if video is None:
            raise ExpertNotFoundError(
                f"Expert video not found for expert_video_id={expert_video_id}."
            )
        if video.video_role != "expert":
            raise ExpertSam2Error(
                f"Video {expert_video_id} is not an expert reference "
                f"(video_role={video.video_role!r})."
            )
    else:
        video = db.execute(
            select(Video).where(
                Video.chapter_id == str(chapter_id),
                Video.video_role == "expert",
            )
        ).scalar_one_or_none()
        if video is None:
            raise ExpertNotFoundError(
                f"No expert video found for chapter_id={chapter_id}."
            )

    chapter: Optional[Chapter] = None
    if video.chapter_id:
        chapter = db.execute(
            select(Chapter).where(Chapter.id == video.chapter_id)
        ).scalar_one_or_none()

    return _ResolvedExpert(video=video, chapter=chapter)


# ---------------------------------------------------------------------------
# Source video handling
# ---------------------------------------------------------------------------

def save_expert_source_video(
    video: Video,
    expert_folder: Path,
    *,
    override_video_path: Optional[str | Path] = None,
) -> Path:
    """Copy the expert source video into the stable SAM 2 folder as ``source.mp4``.

    If the MediaPipe expert flow already placed a ``source.mp4`` under
    ``storage/expert/mediapipe/<expert_code>/``, we use that as the
    default source instead of the raw upload, so the two pipelines see
    the same pixels. Callers can still override with ``override_video_path``.
    """
    if override_video_path is not None:
        source_path = resolve_video_path(override_video_path)
    else:
        mp_source = (
            Path(settings.STORAGE_ROOT)
            / "expert"
            / "mediapipe"
            / str(video.id)
            / "source.mp4"
        )
        if mp_source.is_file():
            source_path = mp_source.resolve()
        elif video.file_path:
            source_path = resolve_video_path(video.file_path)
        else:
            raise ExpertSam2Error(
                f"Expert video {video.id} has no file_path set and no MediaPipe source; "
                "cannot locate source."
            )

    destination = expert_folder / SOURCE_FILENAME
    if destination.exists():
        destination.unlink()
    shutil.copyfile(source_path, destination)
    return destination


# ---------------------------------------------------------------------------
# Metadata / summary parsing
# ---------------------------------------------------------------------------

def load_sam2_metadata(metadata_path: Path) -> SAM2RunMeta:
    """Load ``metadata.json`` produced by a SAM 2 run."""
    if not metadata_path.is_file():
        raise ExpertSam2Error(f"metadata.json not found: {metadata_path}")
    with metadata_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    try:
        return SAM2RunMeta.model_validate(payload)
    except Exception as exc:  # noqa: BLE001 - surface a cleaner message
        raise ExpertSam2Error(
            f"Invalid metadata.json at {metadata_path}: {exc}"
        ) from exc


def load_sam2_summary(summary_path: Path) -> SAM2SummaryDocument:
    """Load ``summary.json`` produced by a SAM 2 run."""
    if not summary_path.is_file():
        raise ExpertSam2Error(f"summary.json not found: {summary_path}")
    with summary_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    try:
        return SAM2SummaryDocument.model_validate(payload)
    except Exception as exc:  # noqa: BLE001
        raise ExpertSam2Error(
            f"Invalid summary.json at {summary_path}: {exc}"
        ) from exc


def _summary_response(
    *,
    metadata: SAM2RunMeta,
    summary: SAM2SummaryDocument,
) -> ExpertSam2Summary:
    return ExpertSam2Summary(
        fps=metadata.fps,
        frame_count=metadata.frame_count,
        width=metadata.width,
        height=metadata.height,
        model_name=metadata.model_name,
        total_frames=summary.total_frames,
        frames_with_mask=summary.frames_with_mask,
        detection_rate=summary.detection_rate,
        mean_mask_area_px=summary.mean_mask_area_px,
        mean_centroid_speed_px_per_frame=summary.mean_centroid_speed_px_per_frame,
        track_fragmentation_count=summary.track_fragmentation_count,
    )


# ---------------------------------------------------------------------------
# DB update
# ---------------------------------------------------------------------------

def update_expert_reference_with_sam2(
    db: Session,
    video: Video,
    *,
    paths: dict[str, Path],
    metadata: SAM2RunMeta,
    summary: SAM2SummaryDocument,
    pipeline_version: Optional[str],
) -> Video:
    """Update the existing expert ``Video`` row with SAM 2 outputs."""
    video.sam2_source_path = _storage_relative(paths["source"])
    video.sam2_raw_path = _storage_relative(paths["raw"])
    video.sam2_summary_path = _storage_relative(paths["summary"])
    video.sam2_metadata_path = _storage_relative(paths["metadata"])
    annotated_path = paths.get("annotated")
    video.sam2_annotated_path = (
        _storage_relative(annotated_path)
        if annotated_path is not None and annotated_path.is_file()
        else None
    )
    video.sam2_status = STATUS_COMPLETED
    video.sam2_processed_at = datetime.now(timezone.utc)
    video.sam2_pipeline_version = pipeline_version or settings.PIPELINE_VERSION
    video.sam2_model_name = metadata.model_name
    video.sam2_fps = Decimal(str(metadata.fps)) if metadata.fps is not None else None
    video.sam2_frame_count = metadata.frame_count
    video.sam2_detection_rate = (
        Decimal(str(summary.detection_rate))
        if summary.detection_rate is not None
        else None
    )

    db.add(video)
    db.commit()
    db.refresh(video)
    return video


def _mark_status(db: Session, video: Video, status: str) -> None:
    """Persist just the SAM 2 status, rolling back any pending changes."""
    try:
        db.rollback()
    except Exception:  # noqa: BLE001 - rollback is best-effort
        pass
    video.sam2_status = status
    db.add(video)
    db.commit()


# ---------------------------------------------------------------------------
# Cleanup
# ---------------------------------------------------------------------------

def cleanup_on_failure(expert_folder: Path, *, keep_source: bool = False) -> None:
    """Remove partial SAM 2 artifacts after a failed run."""
    if not expert_folder.exists():
        return
    for filename in (
        RAW_FILENAME,
        SUMMARY_FILENAME,
        METADATA_FILENAME,
        ANNOTATED_FILENAME,
        INIT_DEBUG_IMAGE_FILENAME,
    ):
        candidate = expert_folder / filename
        if candidate.exists():
            try:
                candidate.unlink()
            except OSError as exc:
                logger.warning(
                    "Failed to remove partial %s in %s: %s",
                    filename,
                    expert_folder,
                    exc,
                )
    if not keep_source:
        source_candidate = expert_folder / SOURCE_FILENAME
        if source_candidate.exists():
            try:
                source_candidate.unlink()
            except OSError as exc:
                logger.warning(
                    "Failed to remove partial source in %s: %s",
                    expert_folder,
                    exc,
                )


def _cleanup_run_dir(run_dir: Path) -> None:
    """Remove the temporary ``storage/sam2/runs/<run_id>`` folder."""
    if run_dir.exists():
        try:
            shutil.rmtree(run_dir)
        except OSError as exc:
            logger.warning("Failed to remove temp SAM 2 run dir %s: %s", run_dir, exc)


# ---------------------------------------------------------------------------
# Reuse / completeness helpers
# ---------------------------------------------------------------------------

def _has_completed_outputs(video: Video, expert_folder: Path) -> bool:
    """Check that the SAM 2 contract outputs exist on disk."""
    if video.sam2_status != STATUS_COMPLETED:
        return False
    required_keys = ("source", "raw", "summary", "metadata")
    paths = _expert_file_paths(expert_folder)
    for key in required_keys:
        if not paths[key].is_file():
            return False
    return True


def _build_reference(
    *,
    video: Video,
    chapter: Optional[Chapter],
    paths: dict[str, Path],
    metadata: SAM2RunMeta,
    summary: SAM2SummaryDocument,
    reused_existing: bool,
) -> ExpertSam2Reference:
    annotated_candidate = paths.get("annotated")
    annotated_path_str: Optional[str] = None
    if annotated_candidate is not None and annotated_candidate.is_file():
        annotated_path_str = _storage_relative(annotated_candidate)

    init_debug_candidate = paths.get("init_debug_image")
    init_debug_path_str: Optional[str] = None
    if init_debug_candidate is not None and init_debug_candidate.is_file():
        init_debug_path_str = _storage_relative(init_debug_candidate)

    return ExpertSam2Reference(
        expert_video_id=UUID(str(video.id)),
        expert_code=str(video.id),
        chapter_id=UUID(str(video.chapter_id)) if video.chapter_id else None,
        title=chapter.title if chapter is not None else None,
        source_path=_storage_relative(paths["source"]),
        raw_path=_storage_relative(paths["raw"]),
        summary_path=_storage_relative(paths["summary"]),
        metadata_path=_storage_relative(paths["metadata"]),
        annotated_path=annotated_path_str,
        init_debug_image_path=init_debug_path_str,
        sam2_status=video.sam2_status or STATUS_COMPLETED,
        sam2_processed_at=video.sam2_processed_at,
        pipeline_version=video.sam2_pipeline_version,
        summary=_summary_response(metadata=metadata, summary=summary),
        reused_existing=reused_existing,
    )


# ---------------------------------------------------------------------------
# Public orchestration
# ---------------------------------------------------------------------------

def register_expert_sam2_reference(
    db: Session,
    *,
    expert_video_id: Optional[str | UUID] = None,
    chapter_id: Optional[str | UUID] = None,
    video_path: Optional[str | Path] = None,
    overwrite: bool = False,
    render_annotation: bool = True,
    pipeline_version: Optional[str] = None,
) -> ExpertSam2Reference:
    """One-time preprocessing of an expert reference video with SAM 2.

    Flow (mirrors ``register_expert_mediapipe_reference``):

        1. Resolve the existing expert ``Video`` row from DB.
        2. If already completed and ``overwrite=False``, return as-is.
        3. Ensure the stable expert SAM 2 folder exists.
        4. Copy the source video into the stable folder as ``source.mp4``.
        5. Run the SAM 2 pipeline seeded from the expert's MediaPipe
           features.json (no UI clicks, ever).
        6. Copy raw/summary/metadata (+ optional annotated) into the
           stable folder.
        7. Parse ``metadata.json`` + ``summary.json`` to extract summary fields.
        8. Update the expert Video row with paths + summary.
        9. Return an ``ExpertSam2Reference`` response.
    """
    resolved = _resolve_expert(
        db, expert_video_id=expert_video_id, chapter_id=chapter_id
    )
    video = resolved.video
    expert_code = str(video.id)
    expert_folder = ensure_expert_sam2_dir(expert_code)

    # Case A: already completed + files exist on disk => reuse.
    if not overwrite and _has_completed_outputs(video, expert_folder):
        paths = _expert_file_paths(expert_folder)
        metadata = load_sam2_metadata(paths["metadata"])
        summary = load_sam2_summary(paths["summary"])
        logger.info("Reusing existing expert SAM 2 reference for %s", expert_code)
        return _build_reference(
            video=video,
            chapter=resolved.chapter,
            paths=paths,
            metadata=metadata,
            summary=summary,
            reused_existing=True,
        )

    # Case B/C: fresh processing (or overwrite / incomplete) — run the pipeline.
    mediapipe_features_path = _resolve_expert_mediapipe_features_path(video)

    _mark_status(db, video, STATUS_PROCESSING)

    if overwrite:
        cleanup_on_failure(expert_folder, keep_source=False)

    temp_run_id = f"expert_sam2_{expert_code}_{uuid4().hex[:8]}"
    temp_run_dir: Optional[Path] = None

    try:
        source_in_expert = save_expert_source_video(
            video,
            expert_folder,
            override_video_path=video_path,
        )

        artifacts: SAM2ContractArtifacts = run_sam2_from_mediapipe_prompt(
            source_in_expert,
            mediapipe_features_path,
            run_id=temp_run_id,
            render_annotation=render_annotation,
        )
        temp_run_dir = artifacts.run_dir

        paths = copy_contract_artifacts_to(
            artifacts,
            target_folder=expert_folder,
            source_video=None,  # already in place
            include_annotated=render_annotation,
        )
        paths["source"] = source_in_expert
        paths.setdefault("annotated", expert_folder / ANNOTATED_FILENAME)
        paths.setdefault("init_debug_image", expert_folder / INIT_DEBUG_IMAGE_FILENAME)

        metadata = load_sam2_metadata(paths["metadata"])
        summary = load_sam2_summary(paths["summary"])

        update_expert_reference_with_sam2(
            db,
            video,
            paths=paths,
            metadata=metadata,
            summary=summary,
            pipeline_version=pipeline_version,
        )
    except SAM2AssetsMissingError as exc:
        logger.exception("SAM 2 assets missing for expert %s", expert_code)
        _mark_status(db, video, STATUS_FAILED)
        cleanup_on_failure(expert_folder, keep_source=True)
        if temp_run_dir is not None:
            _cleanup_run_dir(temp_run_dir)
        raise ExpertSam2AssetsMissingError(
            "SAM2 assets are missing. Required checkpoint/config files were not found.",
            debug=exc.to_debug_payload(),
        ) from exc
    except SAM2GPUOutOfMemoryError as exc:
        logger.exception(
            "SAM 2 GPU out-of-memory for expert %s (device=%s)",
            expert_code,
            exc.resolved_device,
        )
        _mark_status(db, video, STATUS_FAILED)
        cleanup_on_failure(expert_folder, keep_source=True)
        if temp_run_dir is not None:
            _cleanup_run_dir(temp_run_dir)
        raise ExpertSam2GPUOutOfMemoryError(
            str(exc),
            resolved_device=exc.resolved_device,
        ) from exc
    except (SAM2InitError, SAM2DependencyError) as exc:
        logger.exception("SAM 2 initialization failed for expert %s", expert_code)
        _mark_status(db, video, STATUS_FAILED)
        cleanup_on_failure(expert_folder, keep_source=True)
        if temp_run_dir is not None:
            _cleanup_run_dir(temp_run_dir)
        raise ExpertSam2Error(
            f"SAM 2 preprocessing could not initialize for expert {expert_code}: {exc}"
        ) from exc
    except (SAM2PipelineError, SAM2Error) as exc:
        logger.exception("SAM 2 pipeline failed for expert %s", expert_code)
        _mark_status(db, video, STATUS_FAILED)
        cleanup_on_failure(expert_folder, keep_source=True)
        if temp_run_dir is not None:
            _cleanup_run_dir(temp_run_dir)
        raise ExpertSam2Error(
            f"SAM 2 pipeline failed for expert {expert_code}: {exc}"
        ) from exc
    except Exception as exc:  # noqa: BLE001 - surface any unexpected error
        logger.exception("Unexpected failure while processing expert SAM 2 %s", expert_code)
        _mark_status(db, video, STATUS_FAILED)
        cleanup_on_failure(expert_folder, keep_source=True)
        if temp_run_dir is not None:
            _cleanup_run_dir(temp_run_dir)
        raise ExpertSam2Error(
            f"Expert SAM 2 preprocessing failed for {expert_code}: {exc}"
        ) from exc

    if temp_run_dir is not None:
        _cleanup_run_dir(temp_run_dir)

    return _build_reference(
        video=video,
        chapter=resolved.chapter,
        paths=paths,
        metadata=metadata,
        summary=summary,
        reused_existing=False,
    )
