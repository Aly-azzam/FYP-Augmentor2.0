"""High-level SAM 2 pipeline orchestration.

``sam2_service.py`` owns the low-level predictor + mask propagation and
produces rich ``detections.json`` / ``features.json`` artifacts for
debugging. This module sits on top of it and provides the
downstream-facing layer used by the rest of the backend:

    * ``run_sam2_from_mediapipe_prompt`` — the single entry point that
      auto-builds a SAM 2 prompt from a MediaPipe features document,
      runs the pipeline, and emits the stable ``raw.json`` /
      ``summary.json`` / ``metadata.json`` contract artifacts.
    * ``process_learner_sam2`` — learner-runtime convenience wrapper.
      Expert-side orchestration lives in ``expert_sam2_service.py``,
      which consumes ``run_sam2_from_mediapipe_prompt`` under the hood.

No UI, no manual point picking: the only supported learner flow is
"MediaPipe features.json already exists -> derive prompt automatically".
"""

from __future__ import annotations

import json
import logging
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional
from uuid import uuid4

from app.core.config import settings
from app.core.sam2_constants import (
    SAM2_ANNOTATED_FILENAME,
    SAM2_DEFAULT_ANALYSIS_MODE,
    SAM2_DEFAULT_FRAME_STRIDE,
    SAM2_DEFAULT_MAX_SECONDS,
    SAM2_DEFAULT_MODEL_NAME,
    SAM2_DEFAULT_TARGET_OBJECT_ID,
    SAM2_INIT_DEBUG_IMAGE_FILENAME,
    SAM2_METADATA_FILENAME,
    SAM2_PIPELINE_NAME,
    SAM2_RAW_FILENAME,
    SAM2_RUNS_SUBDIR,
    SAM2_SUMMARY_FILENAME,
)
from app.schemas.sam2.sam2_contract_schema import (
    SAM2RawDocument,
    SAM2RawFrame,
    SAM2SummaryDocument,
    SAM2SummaryWorkingRegion,
)
from app.schemas.sam2.sam2_schema import (
    SAM2DetectionsDocument,
    SAM2FeaturesDocument,
    SAM2InitPrompt,
    SAM2RunMeta,
)
from app.services.sam2.sam2_service import (
    SAM2AssetsMissingError,
    SAM2DependencyError,
    SAM2Error,
    SAM2GPUOutOfMemoryError,
    SAM2InitError,
    SAM2RunArtifacts,
    build_init_prompt_from_mediapipe,
    build_sam2_auto_prompt,
    run_sam2_pipeline,
)
from app.utils.sam2.sam2_utils import LANDMARK_ALIAS_HAND_CENTER


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Errors + result types
# ---------------------------------------------------------------------------

class SAM2PipelineError(SAM2Error):
    """Raised when the SAM 2 high-level orchestration fails."""


@dataclass
class SAM2ContractArtifacts:
    """Paths + in-memory contract documents produced by a run.

    This is the stable shape the rest of the backend consumes. The
    underlying rich ``SAM2RunArtifacts`` is also returned for callers
    that want to inspect internal detection / feature details.
    """

    run_id: str
    run_dir: Path
    raw_path: Path
    summary_path: Path
    metadata_path: Path
    annotated_video_path: Optional[Path]
    source_video_path: Path

    raw: SAM2RawDocument
    summary: SAM2SummaryDocument
    metadata: SAM2RunMeta

    underlying: SAM2RunArtifacts
    warnings: List[str] = field(default_factory=list)
    init_debug_image_path: Optional[Path] = None


# ---------------------------------------------------------------------------
# Contract derivation (pure; no I/O)
# ---------------------------------------------------------------------------

def _bbox_to_xyxy(bbox) -> Optional[List[int]]:
    if bbox is None:
        return None
    return [int(bbox.x_min), int(bbox.y_min), int(bbox.x_max), int(bbox.y_max)]


def build_sam2_raw_document(
    *,
    detections: SAM2DetectionsDocument,
    metadata: SAM2RunMeta,
) -> SAM2RawDocument:
    """Project the rich ``detections.json`` down to the stable ``raw.json`` shape."""
    frames: List[SAM2RawFrame] = []
    for detection in detections.frames:
        frames.append(
            SAM2RawFrame(
                frame_index=int(detection.frame_index),
                timestamp_sec=(
                    float(detection.timestamp_sec)
                    if detection.timestamp_sec is not None
                    else None
                ),
                object="learner_local_region",
                has_mask=bool(detection.has_mask),
                mask_bbox_xyxy=_bbox_to_xyxy(detection.mask_bbox),
                mask_centroid_xy=(
                    [float(detection.mask_centroid_xy[0]), float(detection.mask_centroid_xy[1])]
                    if detection.mask_centroid_xy is not None
                    else None
                ),
                mask_area_px=(
                    int(detection.mask_area_px)
                    if detection.mask_area_px is not None
                    else None
                ),
                quality_flag=str(getattr(detection, "quality_flag", "ok")),
                status=str(getattr(detection, "status", getattr(detection, "quality_flag", "ok"))),
                bbox=_bbox_to_xyxy(detection.mask_bbox),
                centroid=(
                    [float(detection.mask_centroid_xy[0]), float(detection.mask_centroid_xy[1])]
                    if detection.mask_centroid_xy is not None
                    else None
                ),
                area=(
                    int(detection.mask_area_px)
                    if detection.mask_area_px is not None
                    else None
                ),
            )
        )

    return SAM2RawDocument(
        run_id=detections.run_id,
        source_video_path=detections.source_video_path,
        pipeline_name=metadata.pipeline_name,
        model_name=metadata.model_name,
        fps=detections.fps,
        frame_count=detections.frame_count,
        width=detections.width,
        height=detections.height,
        target_object_id=int(detections.target_object_id),
        frames=frames,
    )


def _mean_centroid_speed_px_per_frame(raw_frames: List[SAM2RawFrame]) -> Optional[float]:
    """Average pixel displacement between consecutive frames that both have a centroid.

    Region-support signal only (NOT trajectory scoring). Skips frame
    gaps so a short drop-out doesn't create an artificial spike.
    """
    total = 0.0
    count = 0
    prev_xy: Optional[List[float]] = None
    for frame in raw_frames:
        cur = frame.mask_centroid_xy
        if cur is not None and prev_xy is not None:
            dx = float(cur[0]) - float(prev_xy[0])
            dy = float(cur[1]) - float(prev_xy[1])
            total += (dx * dx + dy * dy) ** 0.5
            count += 1
        prev_xy = cur
    if count == 0:
        return None
    return total / count


def _track_fragmentation_count(raw_frames: List[SAM2RawFrame]) -> int:
    """Count ``has_mask=True -> False`` transitions across consecutive frames."""
    transitions = 0
    prev: Optional[bool] = None
    for frame in raw_frames:
        if prev is True and frame.has_mask is False:
            transitions += 1
        prev = frame.has_mask
    return transitions


def build_sam2_summary_document(
    *,
    raw: SAM2RawDocument,
    features: SAM2FeaturesDocument,
    metadata: SAM2RunMeta,
) -> SAM2SummaryDocument:
    """Aggregate a run into the stable ``summary.json`` shape."""
    total = len(raw.frames)
    with_mask = sum(1 for f in raw.frames if f.has_mask)
    rate = (with_mask / total) if total > 0 else 0.0

    working: Optional[SAM2SummaryWorkingRegion] = None
    if features.working_region is not None:
        wr = features.working_region
        working = SAM2SummaryWorkingRegion(
            x_min=int(wr.x_min),
            y_min=int(wr.y_min),
            x_max=int(wr.x_max),
            y_max=int(wr.y_max),
            width=int(wr.width),
            height=int(wr.height),
            padding_px=int(wr.padding_px),
            frame_count_used=int(wr.frame_count_used),
        )

    return SAM2SummaryDocument(
        run_id=raw.run_id,
        pipeline_name=raw.pipeline_name,
        model_name=raw.model_name,
        total_frames=total,
        frames_with_mask=with_mask,
        detection_rate=rate,
        mean_mask_area_px=features.area_stats.mean_px,
        min_mask_area_px=features.area_stats.min_px,
        max_mask_area_px=features.area_stats.max_px,
        std_mask_area_px=features.area_stats.std_px,
        mean_centroid_speed_px_per_frame=_mean_centroid_speed_px_per_frame(raw.frames),
        track_fragmentation_count=_track_fragmentation_count(raw.frames),
        working_region=working,
        warnings=list(metadata.warnings),
    )


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------

def get_runs_root() -> Path:
    """Return ``storage/sam2/runs`` — shared runtime folder for SAM 2 runs.

    Mirrors ``storage/mediapipe/runs`` from the MediaPipe service, which
    is what the existing learner-runtime flow uses.
    """
    return Path(settings.STORAGE_ROOT) / SAM2_RUNS_SUBDIR


def resolve_run_dir(run_id: str) -> Path:
    """Return a run folder under ``storage/sam2/runs/``."""
    if not run_id or "/" in run_id or "\\" in run_id or ".." in run_id:
        raise SAM2PipelineError(f"Invalid run_id: {run_id!r}")
    return get_runs_root() / run_id


def _write_json(path: Path, document) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = document.model_dump(mode="json")
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


# ---------------------------------------------------------------------------
# Public entry points
# ---------------------------------------------------------------------------

def run_sam2_from_mediapipe_prompt(
    video_path: str | Path,
    mediapipe_features_path: str | Path,
    *,
    run_id: Optional[str] = None,
    output_dir: Optional[str | Path] = None,
    image_width: Optional[int] = None,
    image_height: Optional[int] = None,
    preferred_landmark: str = LANDMARK_ALIAS_HAND_CENTER,
    analysis_mode: str = SAM2_DEFAULT_ANALYSIS_MODE,
    max_seconds: float = SAM2_DEFAULT_MAX_SECONDS,
    frame_stride: int = SAM2_DEFAULT_FRAME_STRIDE,
    render_annotation: bool = True,
    target_object_id: int = SAM2_DEFAULT_TARGET_OBJECT_ID,
    device: Optional[str] = None,
) -> SAM2ContractArtifacts:
    """Run SAM 2 end-to-end and emit the ``raw/summary/metadata`` contract.

    Steps:
        1. Auto-build the SAM 2 prompt from ``mediapipe_features_path``
           using ``build_init_prompt_from_mediapipe`` (no manual clicks).
        2. Call ``run_sam2_pipeline`` which writes the rich internal
           ``detections.json`` / ``features.json`` / ``metadata.json``
           + optional ``annotated.mp4`` under ``storage/sam2/runs/<run_id>/``.
        3. Project the in-memory documents onto the stable
           ``SAM2RawDocument`` / ``SAM2SummaryDocument`` contract.
        4. Write ``raw.json`` and ``summary.json`` alongside the other
           artifacts in the run folder.

    ``output_dir`` is optional: when omitted we use
    ``storage/sam2/runs/<run_id>/`` just like the MediaPipe learner
    pipeline writes to ``storage/mediapipe/runs/<run_id>/``. Callers
    that need a different folder (e.g. the expert service that later
    copies artifacts into a stable path) can still control the run id
    through ``run_id`` and pull paths off the returned result.
    """
    effective_run_id = run_id or uuid4().hex

    if output_dir is not None:
        custom_run_dir = Path(output_dir)
        # ``run_sam2_pipeline`` always writes into
        # ``storage/sam2/runs/<run_id>/``; a caller that wants a
        # different folder mirrors our contract files there manually
        # after the run. Keeping the contract strict here prevents
        # surprising side-effects from a mismatched output_dir.
        logger.debug(
            "output_dir=%s is informational; run_sam2_pipeline still "
            "writes under storage/sam2/runs/%s/",
            custom_run_dir,
            effective_run_id,
        )

    init_prompt = build_init_prompt_from_mediapipe(
        mediapipe_features_path,
        image_width=image_width,
        image_height=image_height,
        preferred_landmark=preferred_landmark,
        target_object_id=target_object_id,
    )
    auto_prompt = build_sam2_auto_prompt(
        mediapipe_features_path,
        image_width=image_width,
        image_height=image_height,
        preferred_landmark=preferred_landmark,
    )
    point_xy = auto_prompt["point_xy"]
    box_xyxy = auto_prompt.get("box_xyxy")
    run_context = "expert" if str(effective_run_id).startswith("expert_sam2_") else "learner"
    prompt_debug_payload: Dict[str, Any] = {
        "run_id": effective_run_id,
        "run_context": run_context,
        "frame_index": int(auto_prompt["frame_index"]),
        "timestamp_sec": float(auto_prompt["timestamp_sec"]),
        "point_xy": [float(point_xy[0]), float(point_xy[1])],
        "bbox_xyxy": ([float(v) for v in box_xyxy] if box_xyxy is not None else None),
        "landmark_source": auto_prompt.get("source"),
        "preferred_landmark": preferred_landmark,
        "image_width": int(auto_prompt["image_width"]),
        "image_height": int(auto_prompt["image_height"]),
        "prompt_validation": auto_prompt.get("prompt_validation"),
        "negative_points_xy": auto_prompt.get("negative_points_xy"),
        "sam_roi_bbox_xyxy_px": auto_prompt.get("sam_roi_bbox_xyxy_px"),
        "local_roi_side_px": auto_prompt.get("local_roi_side_px"),
        "selection_diagnostics": auto_prompt.get("selection_diagnostics"),
    }
    logger.info("SAM2 prompt debug: %s", prompt_debug_payload)
    prompt_validation = auto_prompt.get("prompt_validation") or {}
    if (
        not bool(prompt_validation.get("point_inside_bbox", True))
        or bool(prompt_validation.get("bbox_too_large", False))
        or bool(prompt_validation.get("bbox_drifting_background", False))
    ):
        logger.warning("SAM2 prompt validation flags: %s", prompt_validation)

    try:
        underlying = run_sam2_pipeline(
            video_path,
            init_prompt,
            run_id=effective_run_id,
            analysis_mode=analysis_mode,
            max_seconds=max_seconds,
            frame_stride=frame_stride,
            render_annotation=render_annotation,
            device=device,
        )
    except (
        SAM2InitError,
        SAM2AssetsMissingError,
        SAM2DependencyError,
        SAM2GPUOutOfMemoryError,
    ):
        # init-time failures should bubble up unchanged so callers can
        # distinguish "no usable MediaPipe frame" from genuine SAM 2
        # runtime failures. GPU-OOM also bubbles up unchanged so the
        # operator sees the actionable error message instead of a wrapped
        # "SAM 2 pipeline failed for run X: ..." string.
        raise
    except Exception as exc:  # noqa: BLE001 - wrap for cleaner error surface
        raise SAM2PipelineError(
            f"SAM 2 pipeline failed for run {effective_run_id}: {exc}"
        ) from exc

    return _finalize_contract_artifacts(underlying)


def _finalize_contract_artifacts(underlying: SAM2RunArtifacts) -> SAM2ContractArtifacts:
    """Derive + persist raw.json / summary.json next to an existing run."""
    raw_document = build_sam2_raw_document(
        detections=underlying.detections,
        metadata=underlying.metadata,
    )
    summary_document = build_sam2_summary_document(
        raw=raw_document,
        features=underlying.features,
        metadata=underlying.metadata,
    )

    raw_path = underlying.run_dir / SAM2_RAW_FILENAME
    summary_path = underlying.run_dir / SAM2_SUMMARY_FILENAME
    # metadata.json is already written by run_sam2_pipeline; reuse it.
    metadata_path = underlying.metadata_path

    _write_json(raw_path, raw_document)
    _write_json(summary_path, summary_document)

    annotated_path = underlying.annotated_video_path

    return SAM2ContractArtifacts(
        run_id=underlying.run_id,
        run_dir=underlying.run_dir,
        raw_path=raw_path,
        summary_path=summary_path,
        metadata_path=metadata_path,
        annotated_video_path=annotated_path,
        source_video_path=underlying.source_video_path,
        raw=raw_document,
        summary=summary_document,
        metadata=underlying.metadata,
        underlying=underlying,
        warnings=list(underlying.warnings),
        init_debug_image_path=underlying.init_debug_image_path,
    )


def process_learner_sam2(
    video_path: str | Path,
    mediapipe_features_path: str | Path,
    *,
    run_id: Optional[str] = None,
    render_annotation: bool = True,
    device: Optional[str] = None,
) -> SAM2ContractArtifacts:
    """Learner-runtime convenience wrapper for SAM 2.

    The learner compare / evaluation flow should call THIS function (not
    ``run_sam2_pipeline`` directly) so the contract artifacts are always
    written and the prompt is always MediaPipe-derived, never picked by
    a human.

    Outputs land in the shared ``storage/sam2/runs/<run_id>/`` folder,
    matching the MediaPipe learner-runtime convention. Expert-reference
    persistence is intentionally out of scope here — use
    ``register_expert_sam2_reference`` instead.
    """
    return run_sam2_from_mediapipe_prompt(
        video_path,
        mediapipe_features_path,
        run_id=run_id,
        render_annotation=render_annotation,
        device=device,
    )


# ---------------------------------------------------------------------------
# Artifact loading (for inspection API + expert reuse path)
# ---------------------------------------------------------------------------

def load_sam2_contract_artifacts(run_dir: str | Path) -> SAM2ContractArtifacts:
    """Reload a previously-completed SAM 2 run from disk.

    Only the contract artifacts (``raw.json`` / ``summary.json`` /
    ``metadata.json``) are mandatory. The rich internal debug artifacts
    are loaded best-effort and attached to ``underlying`` when present.
    """
    from app.services.sam2.sam2_service import load_run_artifacts  # local import to avoid cycles

    folder = Path(run_dir)
    if not folder.is_dir():
        raise SAM2PipelineError(f"SAM 2 run folder not found: {folder}")

    raw_path = folder / SAM2_RAW_FILENAME
    summary_path = folder / SAM2_SUMMARY_FILENAME
    metadata_path = folder / SAM2_METADATA_FILENAME
    annotated_path = folder / SAM2_ANNOTATED_FILENAME
    init_debug_path = folder / SAM2_INIT_DEBUG_IMAGE_FILENAME

    for path in (raw_path, summary_path, metadata_path):
        if not path.is_file():
            raise SAM2PipelineError(
                f"SAM 2 run at {folder} is missing required artifact: {path.name}"
            )

    with raw_path.open("r", encoding="utf-8") as handle:
        raw_document = SAM2RawDocument.model_validate(json.load(handle))
    with summary_path.open("r", encoding="utf-8") as handle:
        summary_document = SAM2SummaryDocument.model_validate(json.load(handle))
    with metadata_path.open("r", encoding="utf-8") as handle:
        metadata_document = SAM2RunMeta.model_validate(json.load(handle))

    # Best-effort rehydration of the rich debug artifacts (may be absent
    # on stable expert folders — we don't treat that as an error).
    try:
        underlying = load_run_artifacts(metadata_document.run_id)
    except Exception:
        underlying = SAM2RunArtifacts(
            run_id=metadata_document.run_id,
            run_dir=folder,
            detections_path=folder / "detections.json",
            features_path=folder / "features.json",
            metadata_path=metadata_path,
            annotated_video_path=annotated_path if annotated_path.is_file() else None,
            source_video_path=Path(metadata_document.source_video_path),
            detections=SAM2DetectionsDocument(
                run_id=metadata_document.run_id,
                source_video_path=metadata_document.source_video_path,
                fps=metadata_document.fps,
                frame_count=metadata_document.frame_count,
                width=metadata_document.width,
                height=metadata_document.height,
                target_object_id=metadata_document.target_object_id,
                init_prompt=metadata_document.init_prompt,
                frames=[],
            ),
            metadata=metadata_document,
            features=SAM2FeaturesDocument(
                run_id=metadata_document.run_id,
                source_video_path=metadata_document.source_video_path,
                fps=metadata_document.fps,
                frame_count=metadata_document.frame_count,
                width=metadata_document.width,
                height=metadata_document.height,
                target_object_id=metadata_document.target_object_id,
            ),
            warnings=list(metadata_document.warnings),
        )

    # Prefer a debug image colocated in the stable folder (expert runs
    # copy it there via ``copy_contract_artifacts_to``); fall back to
    # whatever the transient run folder still has.
    resolved_init_debug: Optional[Path] = (
        init_debug_path if init_debug_path.is_file() else underlying.init_debug_image_path
    )

    return SAM2ContractArtifacts(
        run_id=metadata_document.run_id,
        run_dir=folder,
        raw_path=raw_path,
        summary_path=summary_path,
        metadata_path=metadata_path,
        annotated_video_path=annotated_path if annotated_path.is_file() else None,
        source_video_path=Path(metadata_document.source_video_path),
        raw=raw_document,
        summary=summary_document,
        metadata=metadata_document,
        underlying=underlying,
        warnings=list(metadata_document.warnings),
        init_debug_image_path=resolved_init_debug,
    )


# ---------------------------------------------------------------------------
# Copy / mirror artifacts (used by the expert service)
# ---------------------------------------------------------------------------

def copy_contract_artifacts_to(
    artifacts: SAM2ContractArtifacts,
    *,
    target_folder: Path,
    source_video: Optional[Path] = None,
    source_filename: str = "source.mp4",
    include_annotated: bool = True,
) -> dict[str, Path]:
    """Copy the ``raw/summary/metadata`` + optional ``annotated/source`` set
    from a runtime run folder into a stable folder (used for expert references).

    Returns the map of artifact name -> final path.
    """
    target_folder = Path(target_folder)
    target_folder.mkdir(parents=True, exist_ok=True)

    raw_target = target_folder / SAM2_RAW_FILENAME
    summary_target = target_folder / SAM2_SUMMARY_FILENAME
    metadata_target = target_folder / SAM2_METADATA_FILENAME
    annotated_target = target_folder / SAM2_ANNOTATED_FILENAME
    init_debug_target = target_folder / SAM2_INIT_DEBUG_IMAGE_FILENAME
    source_target = target_folder / source_filename

    shutil.copyfile(artifacts.raw_path, raw_target)
    shutil.copyfile(artifacts.summary_path, summary_target)
    shutil.copyfile(artifacts.metadata_path, metadata_target)

    result: dict[str, Path] = {
        "raw": raw_target,
        "summary": summary_target,
        "metadata": metadata_target,
    }

    if (
        include_annotated
        and artifacts.annotated_video_path is not None
        and artifacts.annotated_video_path.is_file()
    ):
        shutil.copyfile(artifacts.annotated_video_path, annotated_target)
        result["annotated"] = annotated_target

    if (
        artifacts.init_debug_image_path is not None
        and artifacts.init_debug_image_path.is_file()
    ):
        shutil.copyfile(artifacts.init_debug_image_path, init_debug_target)
        result["init_debug_image"] = init_debug_target

    if source_video is not None:
        if not source_video.is_file():
            raise SAM2PipelineError(
                f"Expected source video not found: {source_video}"
            )
        if source_target.exists():
            source_target.unlink()
        shutil.copyfile(source_video, source_target)
        result["source"] = source_target

    return result
