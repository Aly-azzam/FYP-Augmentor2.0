"""High-level orchestrator for the MediaPipe Hands pipeline.

A single "run" corresponds to one video processed by three services in
order:

    1. extraction_service.run_extraction      -> detections.json + metadata.json
    2. feature_service.run_feature_extraction -> features.json
    3. visualization_service.render_annotated_video -> annotated.mp4

All four files are written to ``backend/storage/mediapipe/runs/<run_id>/``.
This module never talks to the DB; it only owns file system side effects
and the run_id lifecycle.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
from uuid import uuid4

from app.core.config import settings
from app.schemas.mediapipe.mediapipe_schema import MediaPipeRunMeta
from app.services.mediapipe.extraction_service import (
    MediaPipeExtractionError,
    MediaPipeExtractionResult,
    SELECTED_HAND_POLICY_PREFER_RIGHT,
    run_extraction,
)
from app.services.mediapipe.feature_service import (
    DEFAULT_TRAJECTORY_HISTORY_SIZE,
    MediaPipeFeatureError,
    run_feature_extraction,
)
from app.services.mediapipe.visualization_service import (
    MediaPipeVisualizationError,
    render_annotated_video,
)


logger = logging.getLogger(__name__)

ANNOTATED_VIDEO_FILENAME = "annotated.mp4"
DETECTIONS_FILENAME = "detections.json"
FEATURES_FILENAME = "features.json"
METADATA_FILENAME = "metadata.json"


class MediaPipeRunError(RuntimeError):
    """Base error for orchestration failures."""


@dataclass
class MediaPipeRunArtifacts:
    """Paths to every artifact produced by a completed run."""

    run_id: str
    run_dir: Path
    detections_path: Path
    features_path: Path
    metadata_path: Path
    annotated_video_path: Path
    source_video_path: Path
    metadata: MediaPipeRunMeta
    partial_errors: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------

def get_runs_root() -> Path:
    """Return the shared ``backend/storage/mediapipe/runs`` folder."""
    return Path(settings.STORAGE_ROOT) / "mediapipe" / "runs"


def resolve_run_dir(run_id: str) -> Path:
    """Return the run folder for a given ``run_id`` (may or may not exist)."""
    if not run_id or "/" in run_id or "\\" in run_id or ".." in run_id:
        raise MediaPipeRunError(f"Invalid run_id: {run_id!r}")
    return get_runs_root() / run_id


def resolve_video_path(video_path: str | Path) -> Path:
    """Resolve a user-supplied video path to an absolute ``Path`` that exists.

    Accepts three shapes (in priority order):

    1. An absolute path on disk.
    2. A path relative to the current process CWD.
    3. A storage-relative key, with or without a leading ``/storage/`` or
       ``storage/`` prefix (e.g. ``expert/<chapter_id>/<uuid>.mp4``).
    """
    if video_path is None or str(video_path).strip() == "":
        raise MediaPipeRunError("video_path must not be empty.")

    raw = str(video_path).strip()

    # Strip the public storage URL prefix if the caller pasted it in.
    for prefix in ("/storage/", "storage/"):
        if raw.startswith(prefix):
            raw = raw[len(prefix):]
            storage_candidate = (Path(settings.STORAGE_ROOT) / raw).resolve()
            if storage_candidate.is_file():
                return storage_candidate
            break

    candidate = Path(raw)
    if candidate.is_absolute():
        resolved = candidate.resolve()
        if resolved.is_file():
            return resolved
        raise MediaPipeRunError(f"Video file not found: {resolved}")

    cwd_candidate = (Path.cwd() / candidate).resolve()
    if cwd_candidate.is_file():
        return cwd_candidate

    storage_candidate = (Path(settings.STORAGE_ROOT) / candidate).resolve()
    if storage_candidate.is_file():
        return storage_candidate

    raise MediaPipeRunError(
        f"Video file not found (tried cwd and storage root): {video_path}"
    )


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def _load_metadata(run_dir: Path) -> MediaPipeRunMeta:
    metadata_path = run_dir / METADATA_FILENAME
    if not metadata_path.is_file():
        raise MediaPipeRunError(
            f"Run is missing metadata.json: {metadata_path}"
        )
    with metadata_path.open("r", encoding="utf-8") as handle:
        return MediaPipeRunMeta.model_validate(json.load(handle))


def run_pipeline(
    video_path: str | Path,
    *,
    run_id: Optional[str] = None,
    max_num_hands: int = 2,
    min_detection_confidence: float = 0.5,
    min_tracking_confidence: float = 0.5,
    selected_hand_policy: str = SELECTED_HAND_POLICY_PREFER_RIGHT,
    trajectory_history_size: int = DEFAULT_TRAJECTORY_HISTORY_SIZE,
    render_annotation: bool = True,
) -> MediaPipeRunArtifacts:
    """Run extraction → features → visualization for a single video.

    Even when no hand is detected on any frame, the run folder, JSON
    documents, and annotated video are still produced so that downstream
    consumers can react to ``detection_rate == 0`` gracefully.
    """
    effective_video_path = resolve_video_path(video_path)
    effective_run_id = run_id or uuid4().hex

    try:
        extraction_result: MediaPipeExtractionResult = run_extraction(
            effective_video_path,
            run_id=effective_run_id,
            max_num_hands=max_num_hands,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
            selected_hand_policy=selected_hand_policy,
        )
    except MediaPipeExtractionError as exc:
        raise MediaPipeRunError(f"Extraction failed: {exc}") from exc

    run_dir = extraction_result.run_dir
    partial_errors: list[str] = []

    try:
        features_path = run_feature_extraction(
            run_dir,
            trajectory_history_size=trajectory_history_size,
        )
    except MediaPipeFeatureError as exc:
        raise MediaPipeRunError(f"Feature extraction failed: {exc}") from exc

    annotated_path = run_dir / ANNOTATED_VIDEO_FILENAME
    if render_annotation:
        try:
            annotated_path = render_annotated_video(
                run_dir,
                source_video_path=effective_video_path,
                output_filename=ANNOTATED_VIDEO_FILENAME,
            )
        except MediaPipeVisualizationError as exc:
            # We intentionally do NOT fail the whole run when visualization
            # fails (e.g. a codec issue on a given platform). The JSON
            # artifacts are the source of truth; the caller can retry
            # annotation separately.
            logger.warning("Annotation failed for run %s: %s", effective_run_id, exc)
            partial_errors.append(f"annotation_failed: {exc}")

    metadata = extraction_result.metadata

    return MediaPipeRunArtifacts(
        run_id=effective_run_id,
        run_dir=run_dir,
        detections_path=extraction_result.detections_path,
        features_path=features_path,
        metadata_path=extraction_result.metadata_path,
        annotated_video_path=annotated_path,
        source_video_path=effective_video_path,
        metadata=metadata,
        partial_errors=partial_errors,
    )


def load_run_artifacts(run_id: str) -> MediaPipeRunArtifacts:
    """Return artifact paths + metadata for an already-completed run."""
    run_dir = resolve_run_dir(run_id)
    if not run_dir.is_dir():
        raise MediaPipeRunError(f"Run not found: {run_id}")

    metadata = _load_metadata(run_dir)

    detections_path = run_dir / DETECTIONS_FILENAME
    features_path = run_dir / FEATURES_FILENAME
    metadata_path = run_dir / METADATA_FILENAME
    annotated_path = run_dir / ANNOTATED_VIDEO_FILENAME

    missing: list[str] = []
    if not detections_path.is_file():
        missing.append(DETECTIONS_FILENAME)
    if not features_path.is_file():
        missing.append(FEATURES_FILENAME)

    partial_errors: list[str] = []
    if missing:
        partial_errors.append("missing_files: " + ", ".join(missing))
    if not annotated_path.is_file():
        partial_errors.append(f"missing_files: {ANNOTATED_VIDEO_FILENAME}")

    return MediaPipeRunArtifacts(
        run_id=run_id,
        run_dir=run_dir,
        detections_path=detections_path,
        features_path=features_path,
        metadata_path=metadata_path,
        annotated_video_path=annotated_path,
        source_video_path=Path(metadata.source_video_path),
        metadata=metadata,
        partial_errors=partial_errors,
    )
