"""One-time expert MediaPipe preprocessing service.

Expert videos are processed EXACTLY ONCE via this service (either from
the ``app/scripts/process_expert_mediapipe.py`` CLI or the admin-only
route). The outputs are saved into a stable, reusable folder:

    backend/storage/expert/mediapipe/{expert_code}/
        source.mp4
        detections.json
        features.json
        metadata.json

``expert_code`` is the expert ``Video.id`` (UUID string). That is the
canonical expert reference identifier that already exists in PostgreSQL,
so we reuse it as the folder name rather than introducing a parallel
"expert code" concept.

This service:
    * never runs during the learner compare / evaluation flow
    * never creates a parallel expert record in the DB
    * only UPDATES the existing expert ``Video`` row with MediaPipe
      file paths and summary metadata
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
from app.models.chapter import Chapter
from app.models.video import Video
from app.schemas.expert_mediapipe_schema import (
    ExpertMediaPipeReference,
    ExpertMediaPipeSummary,
)
from app.schemas.mediapipe.mediapipe_schema import MediaPipeRunMeta
from app.services.mediapipe.run_service import (
    ANNOTATED_VIDEO_FILENAME,
    DETECTIONS_FILENAME,
    FEATURES_FILENAME,
    METADATA_FILENAME,
    MediaPipeRunError,
    resolve_video_path,
    run_pipeline,
)


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SOURCE_FILENAME = "source.mp4"
ANNOTATED_FILENAME = ANNOTATED_VIDEO_FILENAME  # "annotated.mp4"

STATUS_PENDING = "pending"
STATUS_PROCESSING = "processing"
STATUS_COMPLETED = "completed"
STATUS_FAILED = "failed"


class ExpertMediaPipeError(RuntimeError):
    """Raised when the expert preprocessing flow cannot complete."""


class ExpertNotFoundError(ExpertMediaPipeError):
    """Raised when the requested expert Video row does not exist."""


@dataclass(frozen=True)
class _ResolvedExpert:
    video: Video
    chapter: Optional[Chapter]


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------

def get_expert_mediapipe_root() -> Path:
    """Return the stable ``storage/expert/mediapipe`` folder."""
    return Path(settings.STORAGE_ROOT) / "expert" / "mediapipe"


def ensure_expert_mediapipe_dir(expert_code: str) -> Path:
    """Create (if needed) and return the stable expert folder.

    Raises ``ExpertMediaPipeError`` on unsafe ``expert_code`` values.
    """
    if not expert_code or any(token in expert_code for token in ("/", "\\", "..")):
        raise ExpertMediaPipeError(f"Invalid expert_code: {expert_code!r}")
    folder = get_expert_mediapipe_root() / expert_code
    folder.mkdir(parents=True, exist_ok=True)
    return folder


def _expert_file_paths(expert_folder: Path) -> dict[str, Path]:
    return {
        "source": expert_folder / SOURCE_FILENAME,
        "detections": expert_folder / DETECTIONS_FILENAME,
        "features": expert_folder / FEATURES_FILENAME,
        "metadata": expert_folder / METADATA_FILENAME,
        "annotated": expert_folder / ANNOTATED_FILENAME,
    }


def _storage_relative(path: Path) -> str:
    """Return ``path`` relative to STORAGE_ROOT as a POSIX string."""
    try:
        return path.resolve().relative_to(Path(settings.STORAGE_ROOT).resolve()).as_posix()
    except ValueError:
        # File lives outside the storage root — store the absolute path.
        return str(path.resolve())


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
        raise ExpertMediaPipeError(
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
            raise ExpertMediaPipeError(
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
    """Copy the expert source video into the stable folder as ``source.mp4``.

    When ``override_video_path`` is provided, that file is used instead of
    the path stored on the ``Video`` row. This lets the CLI point at an
    ad-hoc file on disk without first uploading it through the normal API.
    """
    if override_video_path is not None:
        source_path = resolve_video_path(override_video_path)
    else:
        if not video.file_path:
            raise ExpertMediaPipeError(
                f"Expert video {video.id} has no file_path set; cannot locate source."
            )
        source_path = resolve_video_path(video.file_path)

    destination = expert_folder / SOURCE_FILENAME
    if destination.exists():
        destination.unlink()
    shutil.copyfile(source_path, destination)
    return destination


# ---------------------------------------------------------------------------
# Metadata parsing
# ---------------------------------------------------------------------------

def load_mediapipe_metadata(metadata_path: Path) -> MediaPipeRunMeta:
    """Load ``metadata.json`` produced by the MediaPipe run pipeline."""
    if not metadata_path.is_file():
        raise ExpertMediaPipeError(f"metadata.json not found: {metadata_path}")
    with metadata_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    try:
        return MediaPipeRunMeta.model_validate(payload)
    except Exception as exc:  # noqa: BLE001 - surface a cleaner message
        raise ExpertMediaPipeError(
            f"Invalid metadata.json at {metadata_path}: {exc}"
        ) from exc


def _summary_from_metadata(metadata: MediaPipeRunMeta) -> ExpertMediaPipeSummary:
    return ExpertMediaPipeSummary(
        fps=metadata.fps,
        frame_count=metadata.frame_count,
        detection_rate=metadata.detection_rate,
        selected_hand_policy=metadata.selected_hand_policy,
        width=metadata.width,
        height=metadata.height,
        total_frames=metadata.total_frames,
        frames_with_detection=metadata.frames_with_detection,
        right_hand_selected_count=metadata.right_hand_selected_count,
        left_hand_selected_count=metadata.left_hand_selected_count,
    )


# ---------------------------------------------------------------------------
# DB update
# ---------------------------------------------------------------------------

def update_expert_reference_with_mediapipe(
    db: Session,
    video: Video,
    *,
    paths: dict[str, Path],
    metadata: MediaPipeRunMeta,
    pipeline_version: Optional[str],
) -> Video:
    """Update the existing expert ``Video`` row with MediaPipe outputs."""
    video.mediapipe_source_path = _storage_relative(paths["source"])
    video.mediapipe_detections_path = _storage_relative(paths["detections"])
    video.mediapipe_features_path = _storage_relative(paths["features"])
    video.mediapipe_metadata_path = _storage_relative(paths["metadata"])
    annotated_path = paths.get("annotated")
    video.mediapipe_annotated_path = (
        _storage_relative(annotated_path)
        if annotated_path is not None and annotated_path.is_file()
        else None
    )
    video.mediapipe_status = STATUS_COMPLETED
    video.mediapipe_processed_at = datetime.now(timezone.utc)
    video.mediapipe_pipeline_version = pipeline_version or settings.PIPELINE_VERSION
    video.mediapipe_fps = Decimal(str(metadata.fps)) if metadata.fps is not None else None
    video.mediapipe_frame_count = metadata.frame_count
    video.mediapipe_detection_rate = (
        Decimal(str(metadata.detection_rate))
        if metadata.detection_rate is not None
        else None
    )
    video.mediapipe_selected_hand_policy = metadata.selected_hand_policy

    db.add(video)
    db.commit()
    db.refresh(video)
    return video


def _mark_status(db: Session, video: Video, status: str) -> None:
    """Persist just the MediaPipe status, rolling back any pending changes."""
    try:
        db.rollback()
    except Exception:  # noqa: BLE001 - rollback is best-effort
        pass
    video.mediapipe_status = status
    db.add(video)
    db.commit()


# ---------------------------------------------------------------------------
# Cleanup
# ---------------------------------------------------------------------------

def cleanup_on_failure(expert_folder: Path, *, keep_source: bool = False) -> None:
    """Remove partial MediaPipe artifacts after a failed run.

    We keep the folder itself so the stable path stays valid, and
    optionally keep ``source.mp4`` so retries don't have to re-copy it.
    """
    if not expert_folder.exists():
        return
    for filename in (
        DETECTIONS_FILENAME,
        FEATURES_FILENAME,
        METADATA_FILENAME,
        ANNOTATED_FILENAME,
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


def _copy_run_outputs_to_expert_folder(
    *,
    detections_src: Path,
    features_src: Path,
    metadata_src: Path,
    annotated_src: Optional[Path],
    expert_folder: Path,
) -> dict[str, Path]:
    """Copy the MediaPipe artifacts into the stable expert folder.

    The annotated video is best-effort: if the pipeline's visualization
    step failed (``annotated_src`` missing on disk) we silently skip it
    so the JSON artifacts still win.
    """
    targets = _expert_file_paths(expert_folder)
    shutil.copyfile(detections_src, targets["detections"])
    shutil.copyfile(features_src, targets["features"])
    shutil.copyfile(metadata_src, targets["metadata"])
    if annotated_src is not None and annotated_src.is_file():
        shutil.copyfile(annotated_src, targets["annotated"])
    return targets


def _cleanup_run_dir(run_dir: Path) -> None:
    """Remove the temporary ``runs/<run_id>`` folder used while extracting."""
    if run_dir.exists():
        try:
            shutil.rmtree(run_dir)
        except OSError as exc:
            logger.warning("Failed to remove temp run dir %s: %s", run_dir, exc)


# ---------------------------------------------------------------------------
# Reuse / completeness helpers
# ---------------------------------------------------------------------------

def _has_completed_outputs(video: Video, expert_folder: Path) -> bool:
    """Check that the core outputs exist on disk.

    The annotated video is treated as optional — missing it does NOT
    trigger a reprocess, because visualization can fail for codec
    reasons even when the JSON artifacts are valid.
    """
    if video.mediapipe_status != STATUS_COMPLETED:
        return False
    required_keys = ("source", "detections", "features", "metadata")
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
    metadata: MediaPipeRunMeta,
    reused_existing: bool,
) -> ExpertMediaPipeReference:
    annotated_candidate = paths.get("annotated")
    annotated_path: Optional[str] = None
    if annotated_candidate is not None and annotated_candidate.is_file():
        annotated_path = _storage_relative(annotated_candidate)

    return ExpertMediaPipeReference(
        expert_video_id=UUID(str(video.id)),
        expert_code=str(video.id),
        chapter_id=UUID(str(video.chapter_id)) if video.chapter_id else None,
        title=chapter.title if chapter is not None else None,
        source_path=_storage_relative(paths["source"]),
        detections_path=_storage_relative(paths["detections"]),
        features_path=_storage_relative(paths["features"]),
        metadata_path=_storage_relative(paths["metadata"]),
        annotated_path=annotated_path,
        mediapipe_status=video.mediapipe_status or STATUS_COMPLETED,
        mediapipe_processed_at=video.mediapipe_processed_at,
        pipeline_version=video.mediapipe_pipeline_version,
        summary=_summary_from_metadata(metadata),
        reused_existing=reused_existing,
    )


# ---------------------------------------------------------------------------
# Public orchestration
# ---------------------------------------------------------------------------

def register_expert_mediapipe_reference(
    db: Session,
    *,
    expert_video_id: Optional[str | UUID] = None,
    chapter_id: Optional[str | UUID] = None,
    video_path: Optional[str | Path] = None,
    overwrite: bool = False,
    pipeline_version: Optional[str] = None,
    max_num_hands: int = 2,
    min_detection_confidence: float = 0.5,
    min_tracking_confidence: float = 0.5,
) -> ExpertMediaPipeReference:
    """One-time preprocessing of an expert reference video.

    Flow:
        1. Resolve the existing expert ``Video`` row from DB.
        2. If already completed and ``overwrite=False``, return as-is.
        3. Create the stable expert folder.
        4. Copy the source video into the stable folder as ``source.mp4``.
        5. Run the existing MediaPipe ``run_pipeline`` (extraction + features).
        6. Copy detections/features/metadata JSON into the stable folder.
        7. Parse ``metadata.json`` to extract summary fields.
        8. Update the expert Video row with paths + summary.
        9. Return an ``ExpertMediaPipeReference`` response.
    """
    resolved = _resolve_expert(
        db, expert_video_id=expert_video_id, chapter_id=chapter_id
    )
    video = resolved.video
    expert_code = str(video.id)
    expert_folder = ensure_expert_mediapipe_dir(expert_code)

    # Case A: already completed + files exist on disk => reuse.
    if not overwrite and _has_completed_outputs(video, expert_folder):
        paths = _expert_file_paths(expert_folder)
        metadata = load_mediapipe_metadata(paths["metadata"])
        logger.info(
            "Reusing existing expert MediaPipe reference for %s", expert_code
        )
        return _build_reference(
            video=video,
            chapter=resolved.chapter,
            paths=paths,
            metadata=metadata,
            reused_existing=True,
        )

    # Case B/C: fresh processing (or overwrite / incomplete) — run the pipeline.
    _mark_status(db, video, STATUS_PROCESSING)

    # If we're overwriting, proactively drop any stale partial files.
    if overwrite:
        cleanup_on_failure(expert_folder, keep_source=False)

    temp_run_id = f"expert_{expert_code}_{uuid4().hex[:8]}"
    temp_run_dir: Optional[Path] = None

    try:
        source_in_expert = save_expert_source_video(
            video,
            expert_folder,
            override_video_path=video_path,
        )

        artifacts = run_pipeline(
            source_in_expert,
            run_id=temp_run_id,
            max_num_hands=max_num_hands,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
            render_annotation=True,
        )
        temp_run_dir = artifacts.run_dir

        paths = _copy_run_outputs_to_expert_folder(
            detections_src=artifacts.detections_path,
            features_src=artifacts.features_path,
            metadata_src=artifacts.metadata_path,
            annotated_src=artifacts.annotated_video_path,
            expert_folder=expert_folder,
        )
        paths["source"] = source_in_expert

        metadata = load_mediapipe_metadata(paths["metadata"])

        update_expert_reference_with_mediapipe(
            db,
            video,
            paths=paths,
            metadata=metadata,
            pipeline_version=pipeline_version,
        )
    except MediaPipeRunError as exc:
        logger.exception("MediaPipe pipeline failed for expert %s", expert_code)
        _mark_status(db, video, STATUS_FAILED)
        cleanup_on_failure(expert_folder, keep_source=True)
        if temp_run_dir is not None:
            _cleanup_run_dir(temp_run_dir)
        raise ExpertMediaPipeError(
            f"MediaPipe pipeline failed for expert {expert_code}: {exc}"
        ) from exc
    except Exception as exc:
        logger.exception("Unexpected failure while processing expert %s", expert_code)
        _mark_status(db, video, STATUS_FAILED)
        cleanup_on_failure(expert_folder, keep_source=True)
        if temp_run_dir is not None:
            _cleanup_run_dir(temp_run_dir)
        raise ExpertMediaPipeError(
            f"Expert MediaPipe preprocessing failed for {expert_code}: {exc}"
        ) from exc

    if temp_run_dir is not None:
        _cleanup_run_dir(temp_run_dir)

    return _build_reference(
        video=video,
        chapter=resolved.chapter,
        paths=paths,
        metadata=metadata,
        reused_existing=False,
    )
