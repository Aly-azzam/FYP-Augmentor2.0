"""Core SAM 2 backend service.

This module is the clean equivalent of the Old SAM2 demo's
``sam_runner.py`` rewritten for AugMentor 2.0. It mirrors the MediaPipe
service architecture:

    1. tracking (this file, SAM 2 predictor + propagation)
    2. extraction -> ``detections.json`` + ``metadata.json``
    3. features   -> ``features.json`` (working region + area stats)
    4. (optional) annotated ``.mp4`` overlay

Scope is limited to:

    * running SAM 2 on a single video
    * producing learner-local region masks + centroid + bbox + area
    * deriving a ROI / working region for later region-support logic

Explicitly NOT part of this step:

    * expert vs learner orchestration (`expert_sam2_service.py` will
      mirror `expert_mediapipe_service.py` in a later step)
    * any comparison / evaluation plumbing
    * any frontend concerns or human-in-the-loop point picking

Design rule: SAM 2 is initialized automatically from MediaPipe output.
The two TODO markers below (in ``build_init_prompt_from_mediapipe`` and
``run_sam2_pipeline``) are the exact seams where that integration will
be plugged in once the expert MediaPipe reference is wired through.
"""

from __future__ import annotations

import json
import logging
from contextlib import nullcontext
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple
from uuid import uuid4

import numpy as np

from app.core.config import settings
from app.core.sam2_constants import (
    SAM2AssetPaths,
    SAM2_ANNOTATED_FILENAME,
    SAM2_DEFAULT_ANALYSIS_MODE,
    SAM2_DEFAULT_FRAME_STRIDE,
    SAM2_DEFAULT_MAX_SECONDS,
    SAM2_DEFAULT_MODEL_NAME,
    SAM2_DEFAULT_TARGET_OBJECT_ID,
    SAM2_DETECTIONS_FILENAME,
    SAM2_FEATURES_FILENAME,
    SAM2_FRAMES_SUBDIR,
    SAM2_INIT_DEBUG_IMAGE_FILENAME,
    SAM2_LOCAL_ROI_MAX_SIDE_RATIO,
    SAM2_LOCAL_ROI_MIN_SIDE_RATIO,
    SAM2_LOCAL_ROI_SCALE_RATIO,
    SAM2_MASKS_SUBDIR,
    SAM2_METADATA_FILENAME,
    SAM2_PIPELINE_NAME,
    SAM2_PROMPT_AVERAGE_FRAME_COUNT,
    SAM2_PROMPT_BORDER_MARGIN_NORM,
    SAM2_PROMPT_BOX_PADDING_RATIO,
    SAM2_PROMPT_CANDIDATE_POOL_SIZE,
    SAM2_PROMPT_MAX_BBOX_AREA_RATIO,
    SAM2_PROMPT_MIN_BBOX_AREA_RATIO,
    SAM2_PROMPT_POINT_INSIDE_BOX_MARGIN_PX,
    SAM2_PROMPT_SOURCE_MEDIAPIPE,
    SAM2_TEMPORAL_MAX_CENTROID_JUMP_RATIO,
    SAM2_TEMPORAL_MIN_COMPONENT_AREA_RATIO,
    SAM2_TEMPORAL_SMOOTHING_WINDOW,
    SAM2_PROMPT_TYPE_BOX,
    SAM2_PROMPT_TYPE_POINT,
    SAM2_RUNS_SUBDIR,
    SAM2_WORKING_REGION_BBOX_QUANTILE,
    SAM2_WORKING_REGION_PADDING_PX,
    get_sam2_asset_paths as get_canonical_sam2_asset_paths,
)
from app.schemas.sam2.sam2_schema import (
    SAM2AreaStats,
    SAM2BoundingBox,
    SAM2DetectionsDocument,
    SAM2FeaturesDocument,
    SAM2FrameFeatures,
    SAM2FrameMask,
    SAM2InitBox,
    SAM2InitPoint,
    SAM2InitPrompt,
    SAM2RunMeta,
    SAM2WorkingRegion,
)
from app.schemas.mediapipe.mediapipe_schema import MediaPipeFeaturesDocument
from app.utils.sam2.sam2_utils import (
    DEFAULT_LANDMARK_FALLBACK_ORDER,
    LANDMARK_ALIAS_HAND_CENTER,
    LANDMARK_ALIAS_INDEX_TIP,
    ensure_clean_dir,
    extract_frame_range_to_folder,
    get_analysis_window,
    get_binary_mask_for_object,
    get_video_info,
    mask_area,
    mask_bbox,
    mask_centroid,
    mask_to_overlay,
    normalized_to_pixel_coords,
    save_binary_mask_image,
    select_prompt_point_from_mediapipe_frame,
    write_video_from_frames,
)


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Errors and result types
# ---------------------------------------------------------------------------

class SAM2Error(RuntimeError):
    """Base error for the SAM 2 backend."""


class SAM2DependencyError(SAM2Error):
    """Raised when SAM 2 / torch are not installed or the checkpoint is missing."""


class SAM2AssetsMissingError(SAM2DependencyError):
    """Raised when required SAM 2 checkpoint/config assets are missing."""

    def __init__(
        self,
        message: str,
        *,
        resolved_checkpoint_path: Path,
        resolved_config_path: Path,
        checkpoint_exists: bool,
        config_exists: bool,
    ):
        super().__init__(message)
        self.error_type = "SAM2AssetsMissingError"
        self.resolved_checkpoint_path = Path(resolved_checkpoint_path)
        self.resolved_config_path = Path(resolved_config_path)
        self.checkpoint_exists = bool(checkpoint_exists)
        self.config_exists = bool(config_exists)
        missing_paths: list[Path] = []
        if not self.checkpoint_exists:
            missing_paths.append(self.resolved_checkpoint_path)
        if not self.config_exists:
            missing_paths.append(self.resolved_config_path)
        self.missing_paths = [Path(p) for p in missing_paths]

    def to_debug_payload(self) -> dict[str, Any]:
        return {
            "error_type": self.error_type,
            "message": str(self),
            "resolved_checkpoint_path": str(self.resolved_checkpoint_path),
            "resolved_config_path": str(self.resolved_config_path),
            "checkpoint_exists": self.checkpoint_exists,
            "config_exists": self.config_exists,
            "missing_paths": [str(path) for path in self.missing_paths],
        }


class SAM2InitError(SAM2Error):
    """Raised when an initializer prompt cannot be built automatically."""


class SAM2GPUOutOfMemoryError(SAM2Error):
    """Raised when CUDA runs out of memory mid-run.

    Typical cause on the AugMentor stack: the backend ``uvicorn`` process
    keeps a SAM 2 predictor resident on the GPU (module-level singleton
    in :mod:`app.services.sam2.sam2_service`) and a second process
    (e.g. ``python -m app.scripts.process_expert_sam2``) tries to load
    its own predictor on the same GPU. The operator-facing fix is to
    run only one of them at a time, or run the CLI with a free GPU.
    """

    def __init__(self, message: str, *, resolved_device: Optional[str] = None):
        super().__init__(message)
        self.resolved_device = resolved_device


def _is_cuda_oom_exception(exc: BaseException) -> bool:
    """Return True when ``exc`` looks like a CUDA / CUBLAS OOM or
    execution failure caused by GPU memory pressure."""
    if isinstance(exc, MemoryError):
        return True
    text = str(exc).lower()
    markers = (
        "out of memory",
        "cuda out of memory",
        "cublas_status_execution_failed",
        "cudnn_status_execution_failed",
        "cuda error: out of memory",
        "memory allocation failure",
    )
    return any(marker in text for marker in markers)


def reset_video_predictor_cache() -> None:
    """Drop the cached SAM 2 predictor and free its GPU memory.

    Used after a CUDA OOM / execution failure so the next attempt
    doesn't reuse a broken CUDA context.
    """
    global _VIDEO_PREDICTOR, _VIDEO_PREDICTOR_DEVICE
    _VIDEO_PREDICTOR = None
    _VIDEO_PREDICTOR_DEVICE = None
    _maybe_cleanup_cuda()


@dataclass
class SAM2RunArtifacts:
    """Paths + documents produced by a completed SAM 2 run."""

    run_id: str
    run_dir: Path
    detections_path: Path
    features_path: Path
    metadata_path: Path
    annotated_video_path: Optional[Path]
    source_video_path: Path
    detections: SAM2DetectionsDocument
    metadata: SAM2RunMeta
    features: SAM2FeaturesDocument
    warnings: List[str] = field(default_factory=list)
    init_debug_image_path: Optional[Path] = None


# ---------------------------------------------------------------------------
# Lazy predictor singleton (torch / sam2 imported on demand)
# ---------------------------------------------------------------------------

_VIDEO_PREDICTOR: Any = None
_VIDEO_PREDICTOR_DEVICE: Optional[str] = None


def _import_torch():
    """Import torch lazily so unit-testing the schemas doesn't require CUDA."""
    try:
        import torch  # type: ignore
    except ImportError as exc:  # pragma: no cover - torch is required at runtime
        raise SAM2DependencyError(
            "PyTorch is required to run SAM 2. Install torch first."
        ) from exc
    return torch


def _import_build_sam2_video_predictor():
    """Lazy import of the SAM 2 factory function."""
    try:
        from sam2.build_sam import build_sam2_video_predictor  # type: ignore
    except ImportError as exc:  # pragma: no cover - sam2 is required at runtime
        raise SAM2DependencyError(
            "The 'sam2' package is not installed. Install SAM 2 and its "
            "dependencies before running this service."
        ) from exc
    return build_sam2_video_predictor


def resolve_device() -> str:
    """Return ``'cuda'`` when available, otherwise ``'cpu'``."""
    torch = _import_torch()
    return "cuda" if torch.cuda.is_available() else "cpu"


def resolve_device_info(device: Optional[str] = None) -> dict[str, Optional[str]]:
    """Return the effective SAM 2 device and optional GPU name."""
    resolved_device = (device or resolve_device()).strip().lower()
    if resolved_device not in {"cuda", "cpu"}:
        raise SAM2DependencyError(
            f"Unsupported SAM 2 device {resolved_device!r}. Expected 'cuda' or 'cpu'."
        )

    gpu_name: Optional[str] = None
    if resolved_device == "cuda":
        torch = _import_torch()
        if not torch.cuda.is_available():
            raise SAM2DependencyError(
                "SAM2 requested CUDA but torch.cuda.is_available() is False."
            )
        try:
            gpu_name = str(torch.cuda.get_device_name(0))
        except Exception:  # noqa: BLE001
            gpu_name = None

    return {"device": resolved_device, "gpu_name": gpu_name}


def get_sam2_asset_paths(
) -> SAM2AssetPaths:
    """Resolve SAM 2 checkpoint/config paths from canonical backend defaults."""
    return get_canonical_sam2_asset_paths()


def validate_sam2_assets() -> SAM2AssetPaths:
    """Validate that required SAM 2 model assets exist on disk."""
    assets = get_sam2_asset_paths()
    checkpoint_exists = assets.checkpoint_path.is_file()
    config_exists = assets.config_path.is_file()
    if not checkpoint_exists or not config_exists:
        exc = SAM2AssetsMissingError(
            "SAM2 assets are missing. Required checkpoint/config files were not found.",
            resolved_checkpoint_path=assets.checkpoint_path,
            resolved_config_path=assets.config_path,
            checkpoint_exists=checkpoint_exists,
            config_exists=config_exists,
        )
        logger.error("SAM2 asset preflight failed: %s", exc.to_debug_payload())
        raise exc

    logger.info(
        "SAM2 asset preflight passed: %s",
        {
            "resolved_checkpoint_path": str(assets.checkpoint_path),
            "resolved_config_path": str(assets.config_path),
            "checkpoint_exists": checkpoint_exists,
            "config_exists": config_exists,
        },
    )
    return assets


def _maybe_cleanup_cuda() -> None:
    try:
        torch = _import_torch()
    except SAM2DependencyError:
        return
    if not torch.cuda.is_available():
        return
    try:
        torch.cuda.synchronize()
    except Exception:  # noqa: BLE001
        pass
    try:
        torch.cuda.empty_cache()
    except Exception:  # noqa: BLE001
        pass


def get_video_predictor(
    *,
    device: Optional[str] = None,
) -> Any:
    """Return a cached SAM 2 video predictor.

    The predictor is built the first time it's requested and kept in a
    module-level singleton to avoid paying the cold-start cost on every
    video. We enable the same CUDA knobs as the old demo.
    """
    global _VIDEO_PREDICTOR, _VIDEO_PREDICTOR_DEVICE

    device_info = resolve_device_info(device)
    resolved_device = str(device_info["device"])

    if _VIDEO_PREDICTOR is not None and _VIDEO_PREDICTOR_DEVICE == resolved_device:
        return _VIDEO_PREDICTOR

    assets = validate_sam2_assets()

    torch = _import_torch()
    build_sam2_video_predictor = _import_build_sam2_video_predictor()

    if resolved_device == "cuda":
        try:
            torch.backends.cudnn.benchmark = True
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        except Exception as exc:  # noqa: BLE001
            logger.debug("Could not tune CUDA backends: %s", exc)

    if resolved_device == "cuda" and device_info["gpu_name"]:
        logger.info(
            "Building SAM 2 video predictor on device=%s gpu=%s",
            resolved_device,
            device_info["gpu_name"],
        )
    else:
        logger.info("Building SAM 2 video predictor on device=%s", resolved_device)
    predictor = build_sam2_video_predictor(
        str(assets.config_path),
        str(assets.checkpoint_path),
        device=resolved_device,
    )

    _VIDEO_PREDICTOR = predictor
    _VIDEO_PREDICTOR_DEVICE = resolved_device
    return predictor


def _autocast_context(device: str):
    """Enable fp16 autocast on CUDA, no-op on CPU."""
    torch = _import_torch()
    if device == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    return nullcontext()


# ---------------------------------------------------------------------------
# Path helpers (mirror MediaPipe run_service)
# ---------------------------------------------------------------------------

def get_runs_root() -> Path:
    """Return ``storage/sam2/runs``."""
    return Path(settings.STORAGE_ROOT) / SAM2_RUNS_SUBDIR


def resolve_run_dir(run_id: str) -> Path:
    """Return the run folder for a given ``run_id``."""
    if not run_id or "/" in run_id or "\\" in run_id or ".." in run_id:
        raise SAM2Error(f"Invalid run_id: {run_id!r}")
    return get_runs_root() / run_id


def resolve_video_path(video_path: str | Path) -> Path:
    """Resolve a user-supplied video path (absolute / cwd-relative / storage-relative)."""
    if video_path is None or str(video_path).strip() == "":
        raise SAM2Error("video_path must not be empty.")

    raw = str(video_path).strip()

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
        raise SAM2Error(f"Video file not found: {resolved}")

    cwd_candidate = (Path.cwd() / candidate).resolve()
    if cwd_candidate.is_file():
        return cwd_candidate

    storage_candidate = (Path(settings.STORAGE_ROOT) / candidate).resolve()
    if storage_candidate.is_file():
        return storage_candidate

    raise SAM2Error(f"Video file not found (tried cwd and storage root): {video_path}")


def _storage_relative(path: Path) -> str:
    """Return ``path`` relative to ``STORAGE_ROOT`` (POSIX) or absolute string."""
    try:
        return Path(path).resolve().relative_to(
            Path(settings.STORAGE_ROOT).resolve()
        ).as_posix()
    except ValueError:
        return str(Path(path).resolve())


# ---------------------------------------------------------------------------
# MediaPipe -> SAM 2 automatic initializer
# ---------------------------------------------------------------------------
#
# The real FYP pipeline does NOT ask the user to click on the image.
# SAM 2 is initialized automatically from the MediaPipe output that we
# already computed and persisted for every video. The functions below
# read ``features.json`` (and the sibling ``metadata.json`` for width /
# height), walk forward to the first frame where MediaPipe actually saw
# the hand, and return a pixel-space ``(x, y)`` anchor together with
# which landmark was ultimately used (so callers can log index vs
# middle vs fallback).

def load_mediapipe_features(
    mediapipe_features_path: str | Path,
) -> MediaPipeFeaturesDocument:
    """Load and validate a MediaPipe ``features.json`` file from disk."""
    path = Path(mediapipe_features_path)
    if not path.is_file():
        raise SAM2InitError(f"MediaPipe features.json not found: {path}")
    try:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        return MediaPipeFeaturesDocument.model_validate(payload)
    except SAM2InitError:
        raise
    except Exception as exc:  # noqa: BLE001 - surface a cleaner error
        raise SAM2InitError(
            f"Invalid MediaPipe features.json at {path}: {exc}"
        ) from exc


def _load_sibling_metadata(features_path: Path) -> Optional[dict]:
    """Return the parsed ``metadata.json`` sitting next to ``features.json``.

    This is how we recover the image width / height when the caller
    didn't pass them explicitly. Returns ``None`` (not a hard error) if
    the file is missing so callers can surface a friendlier message.
    """
    metadata_path = features_path.parent / "metadata.json"
    if not metadata_path.is_file():
        return None
    try:
        with metadata_path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Could not parse metadata.json at %s: %s", metadata_path, exc)
        return None


def find_first_valid_sam2_prompt(
    features_document: MediaPipeFeaturesDocument,
    *,
    preferred_landmark: str = LANDMARK_ALIAS_HAND_CENTER,
    fallback_order: Sequence[str] = DEFAULT_LANDMARK_FALLBACK_ORDER,
    start_frame_index: int = 0,
) -> Optional[Dict[str, Any]]:
    """Walk the features document forward until a usable frame is found.

    Returns a dict with the selected frame / anchor in *normalized*
    coordinates (pixel conversion is the caller's job because it
    requires the image width/height). Returns ``None`` when no frame in
    the window yields a valid landmark.
    """
    for frame in features_document.frames:
        if int(frame.frame_index) < int(start_frame_index):
            continue

        result = select_prompt_point_from_mediapipe_frame(
            frame,
            preferred_landmark=preferred_landmark,
            fallback_order=fallback_order,
        )
        if result is None:
            continue

        (nx, ny), source_alias = result
        return {
            "frame_index": int(frame.frame_index),
            "timestamp_sec": float(frame.timestamp_sec),
            "normalized_xy": [float(nx), float(ny)],
            "source": source_alias,
            "handedness": getattr(frame, "handedness", None),
            "preferred_landmark": preferred_landmark,
            "used_fallback": source_alias != preferred_landmark,
        }

    return None


def _frame_normalized_bbox(frame: Any) -> Optional[Tuple[float, float, float, float]]:
    """Return the MediaPipe hand bounding box for one frame in normalized coords.

    Falls back to reconstructing the bbox from ``wrist + wrist_relative_landmarks``
    when the pre-computed ``hand_bbox`` is missing (older feature documents).
    Returns ``None`` when the frame has no usable landmarks.
    """
    bbox = getattr(frame, "hand_bbox", None)
    if bbox is not None:
        x_min = float(getattr(bbox, "x_min", 0.0))
        y_min = float(getattr(bbox, "y_min", 0.0))
        x_max = float(getattr(bbox, "x_max", 0.0))
        y_max = float(getattr(bbox, "y_max", 0.0))
        if x_max > x_min and y_max > y_min:
            return x_min, y_min, x_max, y_max

    wrist = getattr(frame, "wrist", None)
    relative = getattr(frame, "wrist_relative_landmarks", None)
    if not wrist or not relative or len(wrist) < 2:
        return None

    wx, wy = float(wrist[0]), float(wrist[1])
    xs: List[float] = [wx]
    ys: List[float] = [wy]
    for offset in relative.values():
        if not offset or len(offset) < 2:
            continue
        xs.append(wx + float(offset[0]))
        ys.append(wy + float(offset[1]))

    if len(xs) < 2:
        return None
    return min(xs), min(ys), max(xs), max(ys)


def collect_first_n_valid_prompt_frames(
    features_document: MediaPipeFeaturesDocument,
    *,
    preferred_landmark: str = LANDMARK_ALIAS_HAND_CENTER,
    fallback_order: Sequence[str] = DEFAULT_LANDMARK_FALLBACK_ORDER,
    start_frame_index: int = 0,
    max_frames: int = SAM2_PROMPT_AVERAGE_FRAME_COUNT,
) -> List[Dict[str, Any]]:
    """Collect up to ``max_frames`` valid MediaPipe prompt candidates.

    For each frame we record the chosen anchor landmark (normalized) and
    the normalized hand bounding box (when MediaPipe produced one). This
    backs both the simple "first N consecutive" averaging path and the
    larger candidate pool consumed by ``select_best_prompt_window``.
    """
    collected: List[Dict[str, Any]] = []
    if max_frames <= 0:
        return collected

    for frame in features_document.frames:
        if int(frame.frame_index) < int(start_frame_index):
            continue

        result = select_prompt_point_from_mediapipe_frame(
            frame,
            preferred_landmark=preferred_landmark,
            fallback_order=fallback_order,
        )
        if result is None:
            continue

        (nx, ny), source_alias = result
        wrist = getattr(frame, "wrist", None)
        hand_center = getattr(frame, "hand_center", None)
        wrist_xy = (
            [float(wrist[0]), float(wrist[1])]
            if wrist is not None and len(wrist) >= 2
            else None
        )
        hand_center_xy = (
            [float(hand_center[0]), float(hand_center[1])]
            if hand_center is not None and len(hand_center) >= 2
            else None
        )
        collected.append(
            {
                "frame_index": int(frame.frame_index),
                "timestamp_sec": float(frame.timestamp_sec),
                "normalized_xy": [float(nx), float(ny)],
                "source": source_alias,
                "handedness": getattr(frame, "handedness", None),
                "normalized_bbox": _frame_normalized_bbox(frame),
                "wrist_xy": wrist_xy,
                "hand_center_xy": hand_center_xy,
            }
        )
        if len(collected) >= int(max_frames):
            break
    return collected


def _score_candidate(
    candidate: Dict[str, Any],
    *,
    preferred_landmark: str,
    min_area_ratio: float,
    max_area_ratio: float,
    border_margin: float,
) -> Dict[str, Any]:
    """Score a single candidate prompt frame.

    Higher is better. Candidates with a usable MediaPipe hand bbox that
    sits comfortably inside the frame, has a reasonable area, and whose
    anchor point lies inside the bbox earn the highest scores. Point-only
    candidates (no bbox) are still usable but scored lower so the
    selector prefers box-backed windows when available.
    """
    nx, ny = float(candidate["normalized_xy"][0]), float(candidate["normalized_xy"][1])
    bbox_n = candidate.get("normalized_bbox")
    source = str(candidate.get("source") or "")

    reasons: List[str] = []
    score = 0.0

    # Strong bonus for matching the preferred landmark — fewer fallbacks
    # means we're closer to the canonical action anchor.
    if source == preferred_landmark:
        score += 1.0
    else:
        reasons.append(f"used_fallback:{source}")

    point_in_bounds = (
        border_margin <= nx <= 1.0 - border_margin
        and border_margin <= ny <= 1.0 - border_margin
    )
    if point_in_bounds:
        score += 0.5
    else:
        reasons.append("anchor_near_border")

    if bbox_n is None:
        reasons.append("no_hand_bbox")
        return {"score": score, "reasons": reasons, "has_bbox": False, "area_ratio": None}

    x_min, y_min, x_max, y_max = (float(v) for v in bbox_n)
    bw = max(0.0, x_max - x_min)
    bh = max(0.0, y_max - y_min)
    area_ratio = bw * bh
    score_delta = 0.0

    if area_ratio <= 0.0:
        reasons.append("degenerate_bbox")
        return {
            "score": score,
            "reasons": reasons,
            "has_bbox": False,
            "area_ratio": 0.0,
        }

    if area_ratio < min_area_ratio:
        reasons.append(f"bbox_too_small:{area_ratio:.3f}<{min_area_ratio:.3f}")
        score_delta -= 1.5
    elif area_ratio > max_area_ratio:
        reasons.append(f"bbox_too_large:{area_ratio:.3f}>{max_area_ratio:.3f}")
        score_delta -= 1.5
    else:
        score_delta += 1.0

    # Border margin: the closer bbox gets to any edge, the less trustworthy
    # the hand detection is. We reward bboxes that sit well inside.
    edge_gap = min(x_min, y_min, 1.0 - x_max, 1.0 - y_max)
    if edge_gap < border_margin:
        reasons.append(f"bbox_clipping_border:{edge_gap:.3f}")
        score_delta -= 1.0
    else:
        score_delta += min(edge_gap, 0.15) * 2.0  # small bonus for comfortable margin

    # Point-inside-bbox: with a bit of tolerance.
    inside_x = (x_min - 0.02) <= nx <= (x_max + 0.02)
    inside_y = (y_min - 0.02) <= ny <= (y_max + 0.02)
    if inside_x and inside_y:
        score_delta += 0.75
    else:
        reasons.append("anchor_outside_bbox")
        score_delta -= 0.5

    score += score_delta
    return {
        "score": score,
        "reasons": reasons,
        "has_bbox": True,
        "area_ratio": area_ratio,
        "edge_gap": edge_gap,
    }


def _augment_candidates_with_scores(
    candidates: Sequence[Dict[str, Any]],
    *,
    preferred_landmark: str,
    min_area_ratio: float,
    max_area_ratio: float,
    border_margin: float,
) -> List[Dict[str, Any]]:
    """Return candidates sorted in their original order, each annotated
    with a ``score`` dict from ``_score_candidate``."""
    enriched: List[Dict[str, Any]] = []
    for candidate in candidates:
        score_info = _score_candidate(
            candidate,
            preferred_landmark=preferred_landmark,
            min_area_ratio=min_area_ratio,
            max_area_ratio=max_area_ratio,
            border_margin=border_margin,
        )
        item = dict(candidate)
        item["score"] = score_info
        enriched.append(item)
    return enriched


def _window_stability_score(window: Sequence[Dict[str, Any]]) -> float:
    """Penalize jitter across a window of candidates.

    Uses the standard deviation of the anchor point and the bbox centers
    across the window. Returns a small bonus (>=0) when the window is
    stable, or a negative value when it jitters a lot.
    """
    if len(window) <= 1:
        return 0.0

    xs = np.asarray([float(c["normalized_xy"][0]) for c in window], dtype=np.float64)
    ys = np.asarray([float(c["normalized_xy"][1]) for c in window], dtype=np.float64)
    # 95th-percentile spread proxy: std is cheap and robust enough.
    point_jitter = float(np.hypot(xs.std(), ys.std()))

    cx: List[float] = []
    cy: List[float] = []
    for c in window:
        bbox_n = c.get("normalized_bbox")
        if bbox_n is None:
            continue
        cx.append((float(bbox_n[0]) + float(bbox_n[2])) * 0.5)
        cy.append((float(bbox_n[1]) + float(bbox_n[3])) * 0.5)
    bbox_jitter = (
        float(np.hypot(np.std(cx), np.std(cy))) if len(cx) >= 2 else 0.0
    )

    jitter = max(point_jitter, bbox_jitter)
    # A window that drifts by less than ~2% of the image is effectively stable.
    if jitter < 0.02:
        return 0.5
    if jitter < 0.05:
        return 0.0
    return -1.0 * min(1.0, (jitter - 0.05) * 10.0)


def _select_best_prompt_window(
    scored_candidates: Sequence[Dict[str, Any]],
    *,
    window_size: int,
) -> Dict[str, Any]:
    """Pick the best contiguous (in candidate order) window of candidates.

    Returns a dict with the chosen ``window`` (list of candidates) and a
    ``diagnostics`` dict describing how the choice was made. Falls back
    to the first usable frame when no multi-frame window scores positive.
    """
    diagnostics: Dict[str, Any] = {
        "pool_size": len(scored_candidates),
        "window_size_requested": int(window_size),
        "window_start_candidate_index": None,
        "window_score": None,
        "window_mean_candidate_score": None,
        "window_stability_bonus": None,
        "rejected_reasons_sample": [],
    }

    if not scored_candidates:
        return {"window": [], "diagnostics": diagnostics}

    window_size = max(1, int(window_size))

    # Collect a small sample of rejection reasons for debugging (first 5).
    for c in scored_candidates[:5]:
        reasons = c.get("score", {}).get("reasons", [])
        if reasons:
            diagnostics["rejected_reasons_sample"].append(
                {"frame_index": int(c["frame_index"]), "reasons": list(reasons)}
            )

    if len(scored_candidates) < window_size:
        # Not enough candidates to form a full window; fall back to the
        # best single-candidate score.
        best_idx = int(
            max(
                range(len(scored_candidates)),
                key=lambda i: float(scored_candidates[i]["score"]["score"]),
            )
        )
        window = [scored_candidates[best_idx]]
        diagnostics["window_start_candidate_index"] = best_idx
        diagnostics["window_score"] = float(window[0]["score"]["score"])
        diagnostics["window_mean_candidate_score"] = diagnostics["window_score"]
        diagnostics["window_stability_bonus"] = 0.0
        diagnostics["note"] = "pool_smaller_than_window_size"
        return {"window": window, "diagnostics": diagnostics}

    best_score = -float("inf")
    best_start = 0
    best_stability = 0.0
    for start in range(0, len(scored_candidates) - window_size + 1):
        window = scored_candidates[start : start + window_size]
        mean_candidate_score = float(
            np.mean([c["score"]["score"] for c in window])
        )
        stability = _window_stability_score(window)
        combined = mean_candidate_score + stability
        if combined > best_score:
            best_score = combined
            best_start = start
            best_stability = stability

    chosen = list(scored_candidates[best_start : best_start + window_size])
    diagnostics["window_start_candidate_index"] = int(best_start)
    diagnostics["window_score"] = float(best_score)
    diagnostics["window_mean_candidate_score"] = float(
        np.mean([c["score"]["score"] for c in chosen])
    )
    diagnostics["window_stability_bonus"] = float(best_stability)
    return {"window": chosen, "diagnostics": diagnostics}


def _resolve_features_image_size(
    features_path: Path,
    image_width: Optional[int],
    image_height: Optional[int],
) -> Tuple[int, int]:
    """Resolve the source-video dimensions for a MediaPipe features.json.

    Priority: explicit arguments > sibling ``metadata.json`` next to
    features.json. Raises ``SAM2InitError`` if nothing is usable.
    """
    w = int(image_width) if image_width is not None else 0
    h = int(image_height) if image_height is not None else 0
    if w <= 0 or h <= 0:
        metadata_payload = _load_sibling_metadata(features_path)
        if metadata_payload is not None:
            w = w or int(metadata_payload.get("width") or 0)
            h = h or int(metadata_payload.get("height") or 0)
    if w <= 0 or h <= 0:
        raise SAM2InitError(
            "Could not resolve image width/height. Pass image_width and "
            "image_height explicitly, or place a metadata.json next to "
            f"features.json at {features_path.parent}."
        )
    return w, h


def _find_prompt_candidate_at_or_after_frame(
    candidates: Sequence[Dict[str, Any]],
    *,
    frame_index: int,
) -> Optional[Dict[str, Any]]:
    """Pick the first candidate whose frame index is >= ``frame_index``."""
    target = int(frame_index)
    for candidate in candidates:
        if int(candidate["frame_index"]) >= target:
            return candidate
    return None


def _collect_mediapipe_geometry(
    prompt_frames: Sequence[Dict[str, Any]],
    ) -> Dict[str, Any]:
    """Collect normalized MediaPipe geometry used to derive a local SAM ROI."""
    anchor_x = [float(f["normalized_xy"][0]) for f in prompt_frames]
    anchor_y = [float(f["normalized_xy"][1]) for f in prompt_frames]
    wrist_x = [float(f["wrist_xy"][0]) for f in prompt_frames if f.get("wrist_xy")]
    wrist_y = [float(f["wrist_xy"][1]) for f in prompt_frames if f.get("wrist_xy")]
    center_x = [float(f["hand_center_xy"][0]) for f in prompt_frames if f.get("hand_center_xy")]
    center_y = [float(f["hand_center_xy"][1]) for f in prompt_frames if f.get("hand_center_xy")]
    bboxes_norm = [f["normalized_bbox"] for f in prompt_frames if f.get("normalized_bbox")]

    return {
        "anchor_xy": [float(np.mean(anchor_x)), float(np.mean(anchor_y))],
        "wrist_xy": (
            [float(np.mean(wrist_x)), float(np.mean(wrist_y))]
            if wrist_x and wrist_y
            else None
        ),
        "hand_center_xy": (
            [float(np.mean(center_x)), float(np.mean(center_y))]
            if center_x and center_y
            else None
        ),
        "bboxes_norm": bboxes_norm,
    }


def _derive_local_sam_roi_from_landmarks(
    *,
    geometry: Dict[str, Any],
    image_width: int,
    image_height: int,
    point_inside_box_margin_px: float = SAM2_PROMPT_POINT_INSIDE_BOX_MARGIN_PX,
) -> Dict[str, Any]:
    """Derive a smaller learner-local SAM ROI from MediaPipe geometry."""
    anchor_nx, anchor_ny = (float(v) for v in geometry["anchor_xy"])
    hand_center_xy = geometry.get("hand_center_xy")
    wrist_xy = geometry.get("wrist_xy")
    bboxes_norm = geometry.get("bboxes_norm") or []

    # Center local ROI at a geometric blend of action anchor and hand center.
    center_nx, center_ny = anchor_nx, anchor_ny
    if hand_center_xy is not None:
        center_nx = 0.65 * anchor_nx + 0.35 * float(hand_center_xy[0])
        center_ny = 0.65 * anchor_ny + 0.35 * float(hand_center_xy[1])
    if wrist_xy is not None:
        # Keep ROI on the action side of the hand by biasing away from wrist.
        center_nx = center_nx + 0.12 * (center_nx - float(wrist_xy[0]))
        center_ny = center_ny + 0.12 * (center_ny - float(wrist_xy[1]))

    center_nx = max(0.0, min(1.0, center_nx))
    center_ny = max(0.0, min(1.0, center_ny))
    center_px, center_py = normalized_to_pixel_coords(
        center_nx, center_ny, image_width, image_height
    )

    short_side = float(min(image_width, image_height))
    min_side_px = max(16.0, float(SAM2_LOCAL_ROI_MIN_SIDE_RATIO) * short_side)
    max_side_px = max(min_side_px, float(SAM2_LOCAL_ROI_MAX_SIDE_RATIO) * short_side)
    local_side_px = min_side_px
    if bboxes_norm:
        x_min_n = float(min(b[0] for b in bboxes_norm))
        y_min_n = float(min(b[1] for b in bboxes_norm))
        x_max_n = float(max(b[2] for b in bboxes_norm))
        y_max_n = float(max(b[3] for b in bboxes_norm))
        hand_w_px = max(0.0, (x_max_n - x_min_n) * float(image_width))
        hand_h_px = max(0.0, (y_max_n - y_min_n) * float(image_height))
        local_side_px = float(SAM2_LOCAL_ROI_SCALE_RATIO) * max(
            min(hand_w_px, hand_h_px), min_side_px
        )
    local_side_px = max(min_side_px, min(max_side_px, local_side_px))
    half = 0.5 * local_side_px

    x_min_px = max(0.0, center_px - half)
    y_min_px = max(0.0, center_py - half)
    x_max_px = min(float(image_width - 1), center_px + half)
    y_max_px = min(float(image_height - 1), center_py + half)
    if x_max_px <= x_min_px:
        x_max_px = min(float(image_width - 1), x_min_px + 1.0)
    if y_max_px <= y_min_px:
        y_max_px = min(float(image_height - 1), y_min_px + 1.0)

    # Keep prompt point inside the local ROI.
    px, py = center_px, center_py
    margin = float(point_inside_box_margin_px)
    px = min(max(px, x_min_px + margin), x_max_px - margin)
    py = min(max(py, y_min_px + margin), y_max_px - margin)

    return {
        "point_xy": [float(px), float(py)],
        "point_normalized_xy": [float(center_nx), float(center_ny)],
        "box_xyxy": [float(x_min_px), float(y_min_px), float(x_max_px), float(y_max_px)],
        "box_source": "mediapipe_local_roi",
        "point_reanchored_to_box_center": False,
        "local_roi_side_px": float(local_side_px),
    }


def _build_negative_prompt_points(
    *,
    box_xyxy: Optional[Sequence[float]],
    image_width: int,
    image_height: int,
) -> List[List[float]]:
    """Build optional negative points outside the active hand/tool box."""
    if box_xyxy is None:
        return []
    x_min, y_min, x_max, y_max = (float(v) for v in box_xyxy)
    pad_x = max(8.0, 0.04 * float(image_width))
    pad_y = max(8.0, 0.04 * float(image_height))
    candidates = [
        [x_min - pad_x, y_min - pad_y],  # background/table corner
        [x_max + pad_x, y_min - pad_y],  # likely paper / cutting line side
        [x_min - pad_x, y_max + pad_y],  # lower background region
    ]
    points: List[List[float]] = []
    for x, y in candidates:
        px = max(0.0, min(float(image_width - 1), float(x)))
        py = max(0.0, min(float(image_height - 1), float(y)))
        points.append([px, py])
    return points


def build_sam2_auto_prompt(
    mediapipe_features_path: str | Path,
    *,
    preferred_landmark: str = LANDMARK_ALIAS_HAND_CENTER,
    fallback_order: Sequence[str] = DEFAULT_LANDMARK_FALLBACK_ORDER,
    start_frame_index: int = 0,
    image_width: Optional[int] = None,
    image_height: Optional[int] = None,
    average_frame_count: int = SAM2_PROMPT_AVERAGE_FRAME_COUNT,
    box_padding_ratio: float = SAM2_PROMPT_BOX_PADDING_RATIO,
    candidate_pool_size: int = SAM2_PROMPT_CANDIDATE_POOL_SIZE,
    min_bbox_area_ratio: float = SAM2_PROMPT_MIN_BBOX_AREA_RATIO,
    max_bbox_area_ratio: float = SAM2_PROMPT_MAX_BBOX_AREA_RATIO,
    border_margin_norm: float = SAM2_PROMPT_BORDER_MARGIN_NORM,
) -> Dict[str, Any]:
    """Automatically derive a SAM 2 prompt from a MediaPipe features.json.

    Learner-local ROI logic:

    1. Collect a pool of up to ``candidate_pool_size`` valid MediaPipe
       frames starting from ``start_frame_index``.
    2. Score every candidate with simple heuristics (bbox area not too
       tiny / not too huge, bbox away from image borders, anchor point
       inside bbox, preferred landmark rather than a fallback).
    3. Build local geometric context from landmarks/hand boxes.
    4. Derive a smaller SAM ROI from that context (see
       ``_derive_local_sam_roi_from_landmarks``), then clamp to frame bounds.
    """
    path = Path(mediapipe_features_path)
    features_document = load_mediapipe_features(path)

    effective_width, effective_height = _resolve_features_image_size(
        path, image_width, image_height
    )

    pool = collect_first_n_valid_prompt_frames(
        features_document,
        preferred_landmark=preferred_landmark,
        fallback_order=fallback_order,
        start_frame_index=start_frame_index,
        max_frames=max(int(average_frame_count), int(candidate_pool_size)),
    )
    if not pool:
        raise SAM2InitError(
            "No MediaPipe frame with a usable hand detection was found "
            f"in features.json at {path}."
        )

    scored_pool = _augment_candidates_with_scores(
        pool,
        preferred_landmark=preferred_landmark,
        min_area_ratio=float(min_bbox_area_ratio),
        max_area_ratio=float(max_bbox_area_ratio),
        border_margin=float(border_margin_norm),
    )

    window_size = max(1, int(average_frame_count))
    chosen_window: List[Dict[str, Any]] = list(scored_pool[:window_size])
    if not chosen_window:
        raise SAM2InitError(
            "No MediaPipe frame with a usable hand detection was found "
            f"in features.json at {path}."
        )

    selection = chosen_window[0]
    mediapipe_geometry = _collect_mediapipe_geometry(chosen_window)
    aggregate = _derive_local_sam_roi_from_landmarks(
        geometry=mediapipe_geometry,
        image_width=effective_width,
        image_height=effective_height,
    )

    used_fallback = selection["source"] != preferred_landmark
    averaged_frame_indices = [int(f["frame_index"]) for f in chosen_window]
    point_xy = aggregate["point_xy"]
    box_xyxy = aggregate["box_xyxy"]
    point_nx, point_ny = (float(v) for v in aggregate["point_normalized_xy"])
    point_inside_box: Optional[bool] = None
    bbox_area_ratio: Optional[float] = None
    bbox_too_large: Optional[bool] = None
    if box_xyxy is not None:
        x_min, y_min, x_max, y_max = (float(v) for v in box_xyxy)
        point_inside_box = (
            x_min <= float(point_xy[0]) <= x_max and y_min <= float(point_xy[1]) <= y_max
        )
        bbox_area_ratio = (
            max(0.0, x_max - x_min) * max(0.0, y_max - y_min)
        ) / float(max(1, effective_width * effective_height))
        bbox_too_large = bool(bbox_area_ratio > 0.55)
    prompt_validation = {
        "point_inside_bbox": point_inside_box,
        "bbox_area_ratio": bbox_area_ratio,
        "bbox_too_large": bbox_too_large,
        "bbox_drifting_background": None,
        "point_inside_local_region": bool(
            selection.get("source") in {LANDMARK_ALIAS_HAND_CENTER, LANDMARK_ALIAS_INDEX_TIP}
        ),
    }
    negative_points_xy = _build_negative_prompt_points(
        box_xyxy=box_xyxy,
        image_width=effective_width,
        image_height=effective_height,
    )

    scored_frame_debug = [
        {
            "frame_index": int(c["frame_index"]),
            "timestamp_sec": float(c["timestamp_sec"]),
            "score": float(c["score"]["score"]),
            "has_bbox": bool(c["score"].get("has_bbox")),
            "area_ratio": (
                float(c["score"]["area_ratio"])
                if c["score"].get("area_ratio") is not None
                else None
            ),
            "source": c["source"],
            "reasons": list(c["score"].get("reasons", [])),
        }
        for c in chosen_window
    ]

    return {
        "frame_index": int(selection["frame_index"]),
        "timestamp_sec": float(selection["timestamp_sec"]),
        "point_xy": [float(point_xy[0]), float(point_xy[1])],
        "box_xyxy": box_xyxy,
        "box_source": aggregate["box_source"],
        "source": selection["source"],
        # Debug / provenance fields (useful in CLI logs + API responses):
        "preferred_landmark": preferred_landmark,
        "used_fallback": used_fallback,
        "handedness": selection.get("handedness"),
        "normalized_xy": [point_nx, point_ny],
        "image_width": effective_width,
        "image_height": effective_height,
        "mediapipe_features_path": str(path),
        "prompt_source": SAM2_PROMPT_SOURCE_MEDIAPIPE,
        "averaged_frame_count": len(chosen_window),
        "averaged_frame_indices": averaged_frame_indices,
        "point_reanchored_to_box_center": bool(
            aggregate.get("point_reanchored_to_box_center", False)
        ),
        "selection_diagnostics": {
            "selection_mode": "first_valid_window",
            "candidate_pool_size": len(scored_pool),
            "window_size": window_size,
            "selected_frame_index": int(selection["frame_index"]),
            "selected_reasons": list(selection.get("score", {}).get("reasons", [])),
        },
        "prompt_validation": prompt_validation,
        "sam_roi_bbox_xyxy_px": box_xyxy,
        "local_roi_side_px": aggregate.get("local_roi_side_px"),
        "negative_points_xy": negative_points_xy,
        "scored_window_frames": scored_frame_debug,
    }


def build_init_prompt_from_mediapipe(
    mediapipe_features_path: str | Path,
    *,
    image_width: Optional[int] = None,
    image_height: Optional[int] = None,
    preferred_landmark: str = LANDMARK_ALIAS_HAND_CENTER,
    fallback_order: Sequence[str] = DEFAULT_LANDMARK_FALLBACK_ORDER,
    start_frame_index: int = 0,
    target_object_id: int = SAM2_DEFAULT_TARGET_OBJECT_ID,
    average_frame_count: int = SAM2_PROMPT_AVERAGE_FRAME_COUNT,
    box_padding_ratio: float = SAM2_PROMPT_BOX_PADDING_RATIO,
    candidate_pool_size: int = SAM2_PROMPT_CANDIDATE_POOL_SIZE,
) -> SAM2InitPrompt:
    """Build a ``SAM2InitPrompt`` from a MediaPipe features.json.

    Builds a learner-local ROI prompt (smaller than full-hand coverage)
    with one positive point and optional negative disambiguation points.
    """
    auto = build_sam2_auto_prompt(
        mediapipe_features_path,
        preferred_landmark=preferred_landmark,
        fallback_order=fallback_order,
        start_frame_index=start_frame_index,
        image_width=image_width,
        image_height=image_height,
        average_frame_count=average_frame_count,
        box_padding_ratio=box_padding_ratio,
        candidate_pool_size=candidate_pool_size,
    )
    px, py = auto["point_xy"]
    init_point = SAM2InitPoint(x=float(px), y=float(py), label=1)

    box_xyxy = auto.get("box_xyxy")
    log_payload: Dict[str, Any] = {
        "mediapipe_features_path": auto["mediapipe_features_path"],
        "frame_index": int(auto["frame_index"]),
        "timestamp_sec": float(auto["timestamp_sec"]),
        "point_xy": [float(px), float(py)],
        "box_xyxy": box_xyxy,
        "box_source": auto.get("box_source"),
        "landmark_source": auto.get("source"),
        "prompt_source": auto.get("prompt_source"),
        "preferred_landmark": preferred_landmark,
        "used_fallback": bool(auto.get("used_fallback")),
        "image_width": int(auto["image_width"]),
        "image_height": int(auto["image_height"]),
        "averaged_frame_count": int(auto["averaged_frame_count"]),
        "averaged_frame_indices": list(auto.get("averaged_frame_indices", [])),
        "point_reanchored_to_box_center": bool(
            auto.get("point_reanchored_to_box_center", False)
        ),
        "selection_diagnostics": auto.get("selection_diagnostics"),
        "prompt_validation": auto.get("prompt_validation"),
        "sam_roi_bbox_xyxy_px": auto.get("sam_roi_bbox_xyxy_px"),
        "local_roi_side_px": auto.get("local_roi_side_px"),
        "negative_points_xy": auto.get("negative_points_xy"),
    }
    logger.info("SAM2 auto-init prompt resolved: %s", log_payload)

    if box_xyxy is not None:
        x_min, y_min, x_max, y_max = (float(v) for v in box_xyxy)
        all_points: List[SAM2InitPoint] = [init_point]
        for neg_xy in auto.get("negative_points_xy") or []:
            all_points.append(
                SAM2InitPoint(x=float(neg_xy[0]), y=float(neg_xy[1]), label=0)
            )
        return SAM2InitPrompt(
            type=SAM2_PROMPT_TYPE_BOX,
            source=SAM2_PROMPT_SOURCE_MEDIAPIPE,
            frame_index=int(auto["frame_index"]),
            target_object_id=int(target_object_id),
            object_type="learner_local_region",
            point=init_point,
            points=all_points,
            box=SAM2InitBox(x_min=x_min, y_min=y_min, x_max=x_max, y_max=y_max),
        )

    return SAM2InitPrompt(
        type=SAM2_PROMPT_TYPE_POINT,
        source=SAM2_PROMPT_SOURCE_MEDIAPIPE,
        frame_index=int(auto["frame_index"]),
        target_object_id=int(target_object_id),
        object_type="learner_local_region",
        point=init_point,
        points=[init_point],
    )


def build_manual_point_prompt(
    *,
    frame_index: int,
    x_px: float,
    y_px: float,
    target_object_id: int = SAM2_DEFAULT_TARGET_OBJECT_ID,
) -> SAM2InitPrompt:
    """Build a manual learner-local-region prompt from a click (debug only).

    This intentionally exists only for offline experiments; the
    production learner flow must NEVER call this path. Runtime prompts
    come from ``build_init_prompt_from_mediapipe``.

    Manual clicks are interpreted as learner-local ROI seeds.
    """
    return SAM2InitPrompt(
        type=SAM2_PROMPT_TYPE_BOX,
        source="manual",
        frame_index=int(frame_index),
        target_object_id=int(target_object_id),
        object_type="learner_local_region",
        point=SAM2InitPoint(x=float(x_px), y=float(y_px), label=1),
        points=[SAM2InitPoint(x=float(x_px), y=float(y_px), label=1)],
        box=SAM2InitBox(
            x_min=max(0.0, float(x_px) - 80.0),
            y_min=max(0.0, float(y_px) - 80.0),
            x_max=float(x_px) + 80.0,
            y_max=float(y_px) + 80.0,
        ),
    )


def build_manual_init_prompt(
    *,
    frame_index: int,
    box_xyxy: Optional[Sequence[float]],
    positive_points: Sequence[Sequence[float]],
    negative_points: Optional[Sequence[Sequence[float]]] = None,
    target_object_id: int = SAM2_DEFAULT_TARGET_OBJECT_ID,
) -> SAM2InitPrompt:
    """Build a one-time manual SAM2 initializer (learner-side frame 0)."""
    points: List[SAM2InitPoint] = []
    for p in positive_points:
        if p is None or len(p) < 2:
            continue
        points.append(SAM2InitPoint(x=float(p[0]), y=float(p[1]), label=1))
    for p in negative_points or []:
        if p is None or len(p) < 2:
            continue
        points.append(SAM2InitPoint(x=float(p[0]), y=float(p[1]), label=0))
    if not points and box_xyxy is None:
        raise SAM2InitError("Manual init prompt requires at least one point or a box.")

    box_model: Optional[SAM2InitBox] = None
    if box_xyxy is not None and len(box_xyxy) >= 4:
        x1, y1, x2, y2 = (float(v) for v in box_xyxy[:4])
        box_model = SAM2InitBox(x_min=x1, y_min=y1, x_max=x2, y_max=y2)

    primary_point = points[0] if points else None
    prompt_type = SAM2_PROMPT_TYPE_BOX if box_model is not None else SAM2_PROMPT_TYPE_POINT
    return SAM2InitPrompt(
        type=prompt_type,
        source="manual",
        frame_index=int(frame_index),
        target_object_id=int(target_object_id),
        object_type="learner_local_region",
        point=primary_point,
        points=points if points else None,
        box=box_model,
    )


# ---------------------------------------------------------------------------
# Tracking internals
# ---------------------------------------------------------------------------

def _prompt_tensors(
    prompt: SAM2InitPrompt, device: str
) -> Tuple[Optional[Any], Optional[Any], Optional[Any]]:
    """Convert a prompt into ``(points, labels, box)`` tensors for SAM 2.

    Any of the three returned values may be ``None`` depending on the
    prompt shape:

        * ``type="point"``: returns ``(points, labels, None)``.
        * ``type="box"`` without a disambiguation point: returns
          ``(None, None, box)``.
        * ``type="box"`` with a positive-click point (what
          ``build_init_prompt_from_mediapipe`` produces): returns
          ``(points, labels, box)`` so SAM 2 can combine them.
    """
    torch = _import_torch()

    prompt_points: List[SAM2InitPoint] = []
    if prompt.points:
        prompt_points.extend(list(prompt.points))
    elif prompt.point is not None:
        prompt_points.append(prompt.point)
    has_point = len(prompt_points) > 0
    has_box = prompt.box is not None

    if not has_point and not has_box:
        raise SAM2InitError(
            "SAM2 init prompt must contain at least a point or a box."
        )

    points_tensor: Optional[Any] = None
    labels_tensor: Optional[Any] = None
    box_tensor: Optional[Any] = None

    if has_point:
        xy = [[float(p.x), float(p.y)] for p in prompt_points]
        labels = [int(p.label) for p in prompt_points]
        if device == "cuda":
            points_tensor = torch.tensor(xy, dtype=torch.float32, device=device)
            labels_tensor = torch.tensor(labels, dtype=torch.int32, device=device)
        else:
            points_tensor = np.asarray(xy, dtype=np.float32)
            labels_tensor = np.asarray(labels, dtype=np.int32)

    if has_box:
        box = prompt.box
        box_xyxy = [
            float(box.x_min),
            float(box.y_min),
            float(box.x_max),
            float(box.y_max),
        ]
        if device == "cuda":
            box_tensor = torch.tensor(box_xyxy, dtype=torch.float32, device=device)
        else:
            box_tensor = np.asarray(box_xyxy, dtype=np.float32)

    return points_tensor, labels_tensor, box_tensor


def _collect_propagation_masks(
    predictor: Any,
    inference_state: Any,
    *,
    start_frame_idx: int,
    target_obj_id: int,
    reverse: bool,
) -> Dict[int, np.ndarray]:
    """Walk the predictor's propagation iterator and decode each mask."""
    collected: Dict[int, np.ndarray] = {}
    try:
        iterator = predictor.propagate_in_video(
            inference_state,
            start_frame_idx=start_frame_idx,
            reverse=reverse,
        )
    except TypeError:
        # Older SAM 2 API didn't accept keyword args.
        iterator = predictor.propagate_in_video(inference_state)

    for out_frame_idx, out_obj_ids, out_mask_logits in iterator:
        binary_mask = get_binary_mask_for_object(
            out_obj_ids,
            out_mask_logits,
            target_obj_id=target_obj_id,
        )
        if binary_mask is None:
            continue
        collected[int(out_frame_idx)] = binary_mask
    return collected


def _run_predictor(
    *,
    predictor: Any,
    frames_folder: Path,
    selected_local_frame_index: int,
    prompt: SAM2InitPrompt,
    device: str,
    warnings: List[str],
) -> Tuple[Dict[int, np.ndarray], Dict[int, str]]:
    """Run forward + backward propagation, returning masks + their source."""
    torch = _import_torch()

    masks_by_local_frame: Dict[int, np.ndarray] = {}
    source_by_local_frame: Dict[int, str] = {}

    points_tensor, labels_tensor, box_tensor = _prompt_tensors(prompt, device)
    target_obj_id = int(prompt.target_object_id)

    add_kwargs: Dict[str, Any] = {"obj_id": target_obj_id}
    if points_tensor is not None and labels_tensor is not None:
        add_kwargs["points"] = points_tensor
        add_kwargs["labels"] = labels_tensor
    if box_tensor is not None:
        add_kwargs["box"] = box_tensor

    with torch.inference_mode():
        with _autocast_context(device):
            forward_state = predictor.init_state(video_path=str(frames_folder))

            _, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
                inference_state=forward_state,
                frame_idx=int(selected_local_frame_index),
                **add_kwargs,
            )
            prompt_mask = get_binary_mask_for_object(
                out_obj_ids, out_mask_logits, target_obj_id=target_obj_id
            )
            if prompt_mask is not None:
                masks_by_local_frame[int(selected_local_frame_index)] = prompt_mask
                source_by_local_frame[int(selected_local_frame_index)] = "prompted"

            forward_masks = _collect_propagation_masks(
                predictor,
                forward_state,
                start_frame_idx=int(selected_local_frame_index),
                target_obj_id=target_obj_id,
                reverse=False,
            )
            for k, v in forward_masks.items():
                masks_by_local_frame.setdefault(k, v)
                source_by_local_frame.setdefault(k, "propagated_forward")

    try:
        with torch.inference_mode():
            with _autocast_context(device):
                reverse_state = predictor.init_state(video_path=str(frames_folder))

                _, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
                    inference_state=reverse_state,
                    frame_idx=int(selected_local_frame_index),
                    **add_kwargs,
                )
                backward_masks = _collect_propagation_masks(
                    predictor,
                    reverse_state,
                    start_frame_idx=int(selected_local_frame_index),
                    target_obj_id=target_obj_id,
                    reverse=True,
                )
                for k, v in backward_masks.items():
                    if k not in masks_by_local_frame:
                        masks_by_local_frame[k] = v
                        source_by_local_frame[k] = "propagated_backward"
    except Exception as exc:  # noqa: BLE001 - propagation failure should not sink the run
        warnings.append(f"reverse_tracking_failed: {exc}")

    if device == "cuda":
        try:
            torch.cuda.synchronize()
        except Exception:  # noqa: BLE001
            pass

    return masks_by_local_frame, source_by_local_frame


# ---------------------------------------------------------------------------
# Output assembly
# ---------------------------------------------------------------------------

def _connected_components(mask: np.ndarray) -> List[np.ndarray]:
    """Return binary connected components from a mask."""
    try:
        import cv2
    except Exception:  # noqa: BLE001
        return [np.asarray(mask).astype(np.uint8)]
    working = np.asarray(mask).astype(np.uint8)
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(working, connectivity=8)
    components: List[np.ndarray] = []
    for label_id in range(1, int(n_labels)):
        area = int(stats[label_id, cv2.CC_STAT_AREA])
        if area <= 0:
            continue
        components.append((labels == label_id).astype(np.uint8))
    return components


def _smooth_value(history: Sequence[float], window: int) -> Optional[float]:
    if not history:
        return None
    tail = list(history[-max(1, int(window)):])
    return float(np.mean(np.asarray(tail, dtype=np.float64)))


def _stabilize_masks_temporally(
    *,
    saved_paths: Sequence[Path],
    masks_by_local_frame: Dict[int, np.ndarray],
) -> Tuple[Dict[int, np.ndarray], Dict[int, str]]:
    """Stabilize SAM2 masks around a coherent learner-local ROI."""
    stable_masks: Dict[int, np.ndarray] = {}
    quality_by_local_frame: Dict[int, str] = {}

    if not saved_paths:
        return stable_masks, quality_by_local_frame

    # We use the first decoded frame for dimensions, then derive adaptive thresholds.
    try:
        import cv2
        first = cv2.imread(str(saved_paths[0]), cv2.IMREAD_GRAYSCALE)
        h, w = (int(first.shape[0]), int(first.shape[1])) if first is not None else (720, 1280)
    except Exception:  # noqa: BLE001
        h, w = 720, 1280

    image_area = max(1.0, float(h * w))
    min_component_area = max(32, int(image_area * float(SAM2_TEMPORAL_MIN_COMPONENT_AREA_RATIO)))
    diag = float((w * w + h * h) ** 0.5)
    max_jump_px = float(SAM2_TEMPORAL_MAX_CENTROID_JUMP_RATIO) * diag

    prev_centroid: Optional[Tuple[float, float]] = None
    prev_mask: Optional[np.ndarray] = None
    area_history: List[float] = []
    centroid_x_history: List[float] = []
    centroid_y_history: List[float] = []

    for local_idx in range(len(saved_paths)):
        raw_mask = masks_by_local_frame.get(local_idx)
        if raw_mask is None or not bool(np.asarray(raw_mask).astype(bool).any()):
            quality_by_local_frame[local_idx] = "missing"
            continue

        components = _connected_components(raw_mask)
        components = [c for c in components if int(mask_area(c)) >= min_component_area]
        if not components:
            quality_by_local_frame[local_idx] = "missing"
            continue

        selected = max(components, key=lambda c: int(mask_area(c)))
        if prev_centroid is not None and len(components) > 1:
            ranked: List[Tuple[float, np.ndarray]] = []
            for comp in components:
                cxy = mask_centroid(comp)
                if cxy is None:
                    continue
                dist = float(np.hypot(float(cxy[0]) - prev_centroid[0], float(cxy[1]) - prev_centroid[1]))
                ranked.append((dist, comp))
            if ranked:
                ranked.sort(key=lambda item: item[0])
                selected = ranked[0][1]

        selected_centroid = mask_centroid(selected)
        selected_area = mask_area(selected)
        if selected_centroid is None:
            quality_by_local_frame[local_idx] = "missing"
            continue

        unstable_jump = False
        if prev_centroid is not None:
            jump = float(
                np.hypot(
                    float(selected_centroid[0]) - prev_centroid[0],
                    float(selected_centroid[1]) - prev_centroid[1],
                )
            )
            unstable_jump = jump > max_jump_px

        if unstable_jump and prev_mask is not None:
            stable_masks[local_idx] = prev_mask.copy()
            quality_by_local_frame[local_idx] = "reused_previous"
            reused_centroid = mask_centroid(prev_mask)
            reused_area = mask_area(prev_mask)
            if reused_centroid is not None:
                centroid_x_history.append(float(reused_centroid[0]))
                centroid_y_history.append(float(reused_centroid[1]))
            area_history.append(float(reused_area))
            continue

        quality_by_local_frame[local_idx] = "unstable" if unstable_jump else "ok"
        centroid_x_history.append(float(selected_centroid[0]))
        centroid_y_history.append(float(selected_centroid[1]))
        area_history.append(float(selected_area))

        smooth_window = int(SAM2_TEMPORAL_SMOOTHING_WINDOW)
        smooth_cx = _smooth_value(centroid_x_history, smooth_window)
        smooth_cy = _smooth_value(centroid_y_history, smooth_window)
        if smooth_cx is not None and smooth_cy is not None:
            prev_centroid = (smooth_cx, smooth_cy)
        else:
            prev_centroid = (float(selected_centroid[0]), float(selected_centroid[1]))

        prev_mask = np.asarray(selected).astype(np.uint8)
        stable_masks[local_idx] = prev_mask

    return stable_masks, quality_by_local_frame

def _build_detections_and_frame_results(
    *,
    saved_paths: List[Path],
    saved_absolute_frame_indices: List[int],
    masks_by_local_frame: Dict[int, np.ndarray],
    source_by_local_frame: Dict[int, str],
    fps: float,
    masks_output_folder: Path,
    target_obj_id: int,
) -> Tuple[List[SAM2FrameMask], List[dict]]:
    """Assemble per-frame SAM 2 records and matching lightweight drawing data."""
    frame_records: List[SAM2FrameMask] = []
    drawing_records: List[dict] = []
    stable_masks, quality_by_local_frame = _stabilize_masks_temporally(
        saved_paths=saved_paths,
        masks_by_local_frame=masks_by_local_frame,
    )

    centroid_x_history: List[float] = []
    centroid_y_history: List[float] = []
    area_history: List[float] = []
    bbox_x1_history: List[float] = []
    bbox_y1_history: List[float] = []
    bbox_x2_history: List[float] = []
    bbox_y2_history: List[float] = []

    for local_idx, _frame_path in enumerate(saved_paths):
        absolute_frame_index = saved_absolute_frame_indices[local_idx]
        timestamp_sec = absolute_frame_index / fps if fps > 0 else None

        mask = stable_masks.get(local_idx)
        has_mask = mask is not None and bool(np.asarray(mask).astype(bool).any())

        area_value: Optional[int] = None
        centroid_value: Optional[List[float]] = None
        bbox_model: Optional[SAM2BoundingBox] = None
        mask_path_str: Optional[str] = None
        temporal_source = source_by_local_frame.get(local_idx, "none")
        quality_flag = quality_by_local_frame.get(local_idx, "missing")

        if has_mask and mask is not None:
            area_value = mask_area(mask)
            centroid = mask_centroid(mask)
            if centroid is not None:
                centroid_x_history.append(float(centroid[0]))
                centroid_y_history.append(float(centroid[1]))
                smooth_window = int(SAM2_TEMPORAL_SMOOTHING_WINDOW)
                smooth_cx = _smooth_value(centroid_x_history, smooth_window)
                smooth_cy = _smooth_value(centroid_y_history, smooth_window)
                if smooth_cx is not None and smooth_cy is not None:
                    centroid_value = [smooth_cx, smooth_cy]
                else:
                    centroid_value = [float(centroid[0]), float(centroid[1])]
            if area_value is not None:
                area_history.append(float(area_value))
                smooth_area = _smooth_value(area_history, int(SAM2_TEMPORAL_SMOOTHING_WINDOW))
                if smooth_area is not None:
                    area_value = int(round(smooth_area))
            bbox_tuple = mask_bbox(mask)
            if bbox_tuple is not None:
                x_min, y_min, x_max, y_max = bbox_tuple
                bbox_x1_history.append(float(x_min))
                bbox_y1_history.append(float(y_min))
                bbox_x2_history.append(float(x_max))
                bbox_y2_history.append(float(y_max))
                smooth_window = int(SAM2_TEMPORAL_SMOOTHING_WINDOW)
                sx1 = _smooth_value(bbox_x1_history, smooth_window)
                sy1 = _smooth_value(bbox_y1_history, smooth_window)
                sx2 = _smooth_value(bbox_x2_history, smooth_window)
                sy2 = _smooth_value(bbox_y2_history, smooth_window)
                if sx1 is not None and sy1 is not None and sx2 is not None and sy2 is not None:
                    x_min, y_min, x_max, y_max = (
                        int(round(sx1)),
                        int(round(sy1)),
                        int(round(sx2)),
                        int(round(sy2)),
                    )
                bbox_model = SAM2BoundingBox(
                    x_min=x_min,
                    y_min=y_min,
                    x_max=x_max,
                    y_max=y_max,
                    width=max(0, x_max - x_min),
                    height=max(0, y_max - y_min),
                )

            mask_filename = f"frame_{absolute_frame_index:06d}_obj_{target_obj_id:03d}.png"
            saved_mask_path = save_binary_mask_image(
                mask, masks_output_folder / mask_filename
            )
            if saved_mask_path is not None:
                mask_path_str = _storage_relative(saved_mask_path)

        frame_records.append(
            SAM2FrameMask(
                frame_index=int(absolute_frame_index),
                local_frame_index=int(local_idx),
                timestamp_sec=float(timestamp_sec) if timestamp_sec is not None else None,
                has_mask=bool(has_mask),
                mask_area_px=area_value,
                mask_centroid_xy=centroid_value,
                mask_bbox=bbox_model,
                mask_path=mask_path_str,
                temporal_source=temporal_source if has_mask else "none",
                quality_flag=quality_flag if has_mask else "missing",
                status=quality_flag if has_mask else "missing",
            )
        )
        drawing_records.append(
            {
                "local_idx": local_idx,
                "has_mask": has_mask,
                "mask": mask if has_mask else None,
                "bbox": bbox_model,
                "centroid_xy": centroid_value,
                "quality_flag": quality_flag if has_mask else "missing",
            }
        )

    return frame_records, drawing_records


def _wrap_label_to_image_width(
    cv2_module: Any,
    text: str,
    *,
    font: int,
    font_scale: float,
    thickness: int,
    max_width_px: int,
) -> List[str]:
    """Split ``text`` into whitespace-aligned lines that each fit ``max_width_px``.

    Keeps the label readable on portrait-orientation frames where the
    full one-liner would otherwise extend past the image right edge.
    """
    words = text.split(" ")
    lines: List[str] = []
    current = ""
    for word in words:
        candidate = word if not current else f"{current} {word}"
        (tw, _), _ = cv2_module.getTextSize(candidate, font, font_scale, thickness)
        if tw <= max_width_px or not current:
            current = candidate
        else:
            lines.append(current)
            current = word
    if current:
        lines.append(current)
    return lines


def _save_init_prompt_debug_image(
    *,
    frame_path: Path,
    prompt: SAM2InitPrompt,
    output_path: Path,
    label: Optional[str] = None,
) -> Optional[Path]:
    """Draw the point and/or box on the init frame and save it.

    An optional ``label`` is rendered into the top-left corner so an
    operator viewing ``init_prompt_debug.png`` can read frame / prompt
    info without digging into metadata.json. The label is wrapped to
    fit the image width so we never overflow past the right edge (which
    caused ``cv2.imwrite`` to fail on some OpenCV builds with portrait
    frames).

    Best-effort: returns ``None`` (with a warning) when OpenCV cannot
    read or write. The rest of the pipeline does not depend on this
    debug artifact, and label rendering is isolated so a drawing error
    never blocks the underlying annotated image from being saved.
    """
    try:
        import cv2  # local import so tests can skip rendering paths
    except Exception as exc:  # noqa: BLE001
        logger.warning("Could not import cv2 for init debug image: %s", exc)
        return None

    frame = cv2.imread(str(frame_path))
    if frame is None:
        logger.warning("Could not read init frame for debug image: %s", frame_path)
        return None

    image_height, image_width = frame.shape[:2]
    annotated = frame.copy()

    if prompt.box is not None:
        x_min = max(0, min(image_width - 1, int(round(float(prompt.box.x_min)))))
        y_min = max(0, min(image_height - 1, int(round(float(prompt.box.y_min)))))
        x_max = max(0, min(image_width - 1, int(round(float(prompt.box.x_max)))))
        y_max = max(0, min(image_height - 1, int(round(float(prompt.box.y_max)))))
        if x_max > x_min and y_max > y_min:
            cv2.rectangle(
                annotated,
                (x_min, y_min),
                (x_max, y_max),
                color=(0, 255, 255),  # yellow box
                thickness=2,
            )

    points_to_draw: List[SAM2InitPoint] = []
    if prompt.points:
        points_to_draw.extend(prompt.points)
    elif prompt.point is not None:
        points_to_draw.append(prompt.point)
    for point in points_to_draw:
        px = max(0, min(image_width - 1, int(round(float(point.x)))))
        py = max(0, min(image_height - 1, int(round(float(point.y)))))
        color = (0, 255, 0) if int(point.label) == 1 else (0, 0, 255)
        cv2.circle(annotated, (px, py), radius=8, color=(0, 0, 0), thickness=2)
        cv2.circle(annotated, (px, py), radius=5, color=color, thickness=-1)

    if label:
        try:
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.5
            thickness = 1
            line_gap = 4

            # Wrap so the longest line fits inside the frame with a
            # comfortable 16 px margin on each side.
            max_text_width = max(64, int(image_width) - 32)
            lines = _wrap_label_to_image_width(
                cv2,
                str(label),
                font=font,
                font_scale=font_scale,
                thickness=thickness,
                max_width_px=max_text_width,
            )

            if lines:
                line_sizes = [
                    cv2.getTextSize(line, font, font_scale, thickness)[0]
                    for line in lines
                ]
                line_height = max(size[1] for size in line_sizes)
                block_height = (
                    len(lines) * line_height + (len(lines) - 1) * line_gap
                )
                block_width = max(size[0] for size in line_sizes)

                bg_x1 = 6
                bg_y1 = 6
                bg_x2 = min(image_width - 1, bg_x1 + block_width + 8)
                bg_y2 = min(image_height - 1, bg_y1 + block_height + 8)

                # Guard: only draw when clamp still leaves a valid rect.
                if bg_x2 > bg_x1 and bg_y2 > bg_y1:
                    cv2.rectangle(
                        annotated,
                        (bg_x1, bg_y1),
                        (bg_x2, bg_y2),
                        color=(0, 0, 0),
                        thickness=-1,
                    )
                    baseline_y = bg_y1 + line_height + 4
                    for line in lines:
                        cv2.putText(
                            annotated,
                            line,
                            (bg_x1 + 4, baseline_y),
                            font,
                            font_scale,
                            color=(0, 255, 255),
                            thickness=thickness,
                            lineType=cv2.LINE_AA,
                        )
                        baseline_y += line_height + line_gap
        except Exception as exc:  # noqa: BLE001 - label is purely decorative
            logger.warning(
                "Could not render init debug label (%s); saving without label.",
                exc,
            )

    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        ok = cv2.imwrite(str(output_path), annotated)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Could not write init debug image %s: %s", output_path, exc)
        return None

    if not ok:
        # Last-resort: try again without any annotations, in case a
        # platform-specific quirk in libpng rejected the overlay buffer.
        try:
            fallback = frame  # untouched original frame
            ok = cv2.imwrite(str(output_path), fallback)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Could not write unannotated fallback debug image %s: %s",
                output_path,
                exc,
            )
            return None
        if not ok:
            logger.warning(
                "cv2.imwrite returned False for init debug image: %s", output_path
            )
            return None
        logger.warning(
            "cv2.imwrite failed for annotated debug image; saved unannotated "
            "fallback to %s",
            output_path,
        )
    return output_path


def _render_annotated_video(
    *,
    saved_paths: List[Path],
    drawing_records: List[dict],
    output_video_path: Path,
    fps: float,
    overlay_folder: Path,
) -> Optional[Path]:
    """Render learner-local-region overlay with bbox + centroid (best effort)."""
    import cv2  # local import so unit tests can skip this path

    ensure_clean_dir(overlay_folder)
    overlay_paths: List[Path] = []

    for local_idx, frame_path in enumerate(saved_paths):
        frame = cv2.imread(str(frame_path))
        if frame is None:
            continue

        info = drawing_records[local_idx] if local_idx < len(drawing_records) else None
        if info is not None and info["has_mask"] and info["mask"] is not None:
            overlay = mask_to_overlay(frame, info["mask"])
            bbox = info.get("bbox")
            if bbox is not None:
                cv2.rectangle(
                    overlay,
                    (int(bbox.x_min), int(bbox.y_min)),
                    (int(bbox.x_max), int(bbox.y_max)),
                    (255, 255, 0),
                    2,
                )
            centroid_xy = info.get("centroid_xy")
            if centroid_xy is not None and len(centroid_xy) == 2:
                cx = int(round(float(centroid_xy[0])))
                cy = int(round(float(centroid_xy[1])))
                cv2.circle(overlay, (cx, cy), radius=5, color=(0, 255, 255), thickness=-1)
            quality = str(info.get("quality_flag") or "ok")
            cv2.putText(
                overlay,
                f"SAM2 Learner Local Region [{quality}]",
                (12, 24),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 255),
                2,
                cv2.LINE_AA,
            )
        else:
            overlay = frame

        overlay_path = overlay_folder / f"overlay_{local_idx:06d}.jpg"
        cv2.imwrite(str(overlay_path), overlay)
        overlay_paths.append(overlay_path)

    return write_video_from_frames(
        overlay_paths,
        output_video_path,
        fps=fps,
    )


def _compute_working_region(
    frame_records: List[SAM2FrameMask],
    *,
    image_width: int,
    image_height: int,
    padding_px: int = SAM2_WORKING_REGION_PADDING_PX,
    quantile: float = SAM2_WORKING_REGION_BBOX_QUANTILE,
) -> Tuple[Optional[SAM2WorkingRegion], SAM2AreaStats]:
    """Aggregate per-frame bboxes into a single stable working rectangle."""
    bboxes = [fr.mask_bbox for fr in frame_records if fr.mask_bbox is not None]
    areas = [fr.mask_area_px for fr in frame_records if fr.mask_area_px is not None]

    area_stats = SAM2AreaStats()
    if areas:
        area_array = np.asarray(areas, dtype=np.float64)
        area_stats = SAM2AreaStats(
            min_px=int(area_array.min()),
            max_px=int(area_array.max()),
            mean_px=float(area_array.mean()),
            std_px=float(area_array.std(ddof=0)),
        )

    if not bboxes:
        return None, area_stats

    x_min_arr = np.asarray([b.x_min for b in bboxes])
    y_min_arr = np.asarray([b.y_min for b in bboxes])
    x_max_arr = np.asarray([b.x_max for b in bboxes])
    y_max_arr = np.asarray([b.y_max for b in bboxes])

    # Robust aggregate: low quantile of mins, high quantile of maxs,
    # then pad. This is forgiving of a few noisy outlier frames.
    low_q = max(0.0, 1.0 - float(quantile))
    high_q = float(quantile)

    x_min = int(np.quantile(x_min_arr, low_q))
    y_min = int(np.quantile(y_min_arr, low_q))
    x_max = int(np.quantile(x_max_arr, high_q))
    y_max = int(np.quantile(y_max_arr, high_q))

    x_min = max(0, x_min - padding_px)
    y_min = max(0, y_min - padding_px)
    x_max = min(max(0, image_width - 1), x_max + padding_px)
    y_max = min(max(0, image_height - 1), y_max + padding_px)

    if x_max < x_min or y_max < y_min:
        return None, area_stats

    return (
        SAM2WorkingRegion(
            x_min=x_min,
            y_min=y_min,
            x_max=x_max,
            y_max=y_max,
            width=max(0, x_max - x_min),
            height=max(0, y_max - y_min),
            source="quantile",
            quantile=float(quantile),
            padding_px=int(padding_px),
            frame_count_used=len(bboxes),
        ),
        area_stats,
    )


def _build_features_document(
    *,
    run_id: str,
    source_video_path: Path,
    fps: float,
    frame_count: int,
    image_width: int,
    image_height: int,
    target_object_id: int,
    frame_records: List[SAM2FrameMask],
) -> SAM2FeaturesDocument:
    """Derive feature entries (centroid + bbox + optional velocity) + ROI."""
    feature_frames: List[SAM2FrameFeatures] = []
    previous_centroid: Optional[Tuple[float, float]] = None
    previous_timestamp: Optional[float] = None

    for record in frame_records:
        centroid_tuple: Optional[Tuple[float, float]] = None
        if record.mask_centroid_xy is not None:
            centroid_tuple = (
                float(record.mask_centroid_xy[0]),
                float(record.mask_centroid_xy[1]),
            )

        velocity: Optional[List[float]] = None
        if (
            centroid_tuple is not None
            and previous_centroid is not None
            and record.timestamp_sec is not None
            and previous_timestamp is not None
            and record.timestamp_sec > previous_timestamp
        ):
            dt = record.timestamp_sec - previous_timestamp
            velocity = [
                (centroid_tuple[0] - previous_centroid[0]) / dt,
                (centroid_tuple[1] - previous_centroid[1]) / dt,
            ]

        feature_frames.append(
            SAM2FrameFeatures(
                frame_index=record.frame_index,
                timestamp_sec=record.timestamp_sec,
                has_mask=record.has_mask,
                mask_area_px=record.mask_area_px,
                mask_centroid_xy=record.mask_centroid_xy,
                mask_bbox=record.mask_bbox,
                centroid_velocity_px_per_sec=velocity,
            )
        )

        if centroid_tuple is not None and record.timestamp_sec is not None:
            previous_centroid = centroid_tuple
            previous_timestamp = record.timestamp_sec

    working_region, area_stats = _compute_working_region(
        frame_records,
        image_width=image_width,
        image_height=image_height,
    )

    return SAM2FeaturesDocument(
        run_id=run_id,
        source_video_path=str(source_video_path),
        fps=fps,
        frame_count=frame_count,
        width=image_width,
        height=image_height,
        target_object_id=target_object_id,
        working_region=working_region,
        area_stats=area_stats,
        frames=feature_frames,
    )


def _write_json(path: Path, document: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = document.model_dump(mode="json")
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


# ---------------------------------------------------------------------------
# Public entry points
# ---------------------------------------------------------------------------

def run_sam2_pipeline(
    video_path: str | Path,
    init_prompt: SAM2InitPrompt,
    *,
    run_id: Optional[str] = None,
    analysis_mode: str = SAM2_DEFAULT_ANALYSIS_MODE,
    max_seconds: float = SAM2_DEFAULT_MAX_SECONDS,
    frame_stride: int = SAM2_DEFAULT_FRAME_STRIDE,
    render_annotation: bool = True,
    device: Optional[str] = None,
) -> SAM2RunArtifacts:
    """Run SAM 2 on a single video and persist all artifacts.

    Steps:
        1. Probe the video (fps, width, height).
        2. Extract the analysis window to a frames folder.
        3. Build / cache the SAM 2 video predictor.
        4. Run forward + backward propagation from ``init_prompt``.
        5. Build per-frame records + (optional) annotated mp4.
        6. Derive working region + area stats into ``features.json``.
        7. Write ``detections.json`` / ``features.json`` / ``metadata.json``.

    TODO (MediaPipe integration): callers should construct
    ``init_prompt`` via ``build_init_prompt_from_mediapipe`` once the
    expert/learner orchestration layer is introduced. Today the service
    accepts any ``SAM2InitPrompt`` so it can be tested with a manual
    prompt, but the learner pipeline must never expose a UI-driven path.
    """
    warnings: List[str] = []

    resolved_video_path = resolve_video_path(video_path)
    effective_run_id = run_id or uuid4().hex
    run_dir = resolve_run_dir(effective_run_id)
    run_dir.mkdir(parents=True, exist_ok=True)

    info = get_video_info(resolved_video_path)
    if "error" in info:
        raise SAM2Error(f"Could not read video metadata: {info['error']}")
    fps = float(info["fps"])
    width = int(info["width"])
    height = int(info["height"])

    window = get_analysis_window(
        resolved_video_path,
        analysis_mode=analysis_mode,
        max_seconds=max_seconds,
    )
    if "error" in window:
        raise SAM2Error(f"Could not resolve analysis window: {window['error']}")

    frames_folder = run_dir / SAM2_FRAMES_SUBDIR
    extraction = extract_frame_range_to_folder(
        resolved_video_path,
        frames_folder,
        start_frame_index=window["start_frame_index"],
        end_frame_index=window["end_frame_index"],
        frame_stride=frame_stride,
    )
    if "error" in extraction:
        raise SAM2Error(f"Frame extraction failed: {extraction['error']}")

    saved_paths: List[Path] = extraction["saved_paths"]
    saved_absolute_frame_indices: List[int] = extraction["saved_absolute_frame_indices"]
    if not saved_paths:
        raise SAM2Error("No frames extracted for the requested analysis window.")

    # Map the prompt's absolute frame index onto the local stride-aware index.
    try:
        selected_local_frame_index = saved_absolute_frame_indices.index(
            int(init_prompt.frame_index)
        )
    except ValueError:
        # Snap to the closest extracted frame so the caller doesn't have to
        # know about frame_stride.
        selected_local_frame_index = int(
            np.argmin(
                np.abs(
                    np.asarray(saved_absolute_frame_indices) - int(init_prompt.frame_index)
                )
            )
        )
        warnings.append(
            "prompt_frame_index_snapped_to_extracted_frame: "
            f"{init_prompt.frame_index} -> {saved_absolute_frame_indices[selected_local_frame_index]}"
        )

    device_info = resolve_device_info(device)
    resolved_device = str(device_info["device"])
    gpu_name = device_info["gpu_name"]
    assets = validate_sam2_assets()

    if resolved_device == "cuda" and gpu_name:
        logger.info("SAM2 running on: cuda (GPU: %s)", gpu_name)
    else:
        logger.info("SAM2 running on: %s", resolved_device)

    init_debug_image_path: Optional[Path] = None
    try:
        init_frame_path = saved_paths[int(selected_local_frame_index)]
        point_desc = (
            f"point=({int(round(init_prompt.point.x))},{int(round(init_prompt.point.y))})"
            if init_prompt.point is not None
            else "point=None"
        )
        box_desc = (
            "box=("
            f"{int(round(init_prompt.box.x_min))},{int(round(init_prompt.box.y_min))},"
            f"{int(round(init_prompt.box.x_max))},{int(round(init_prompt.box.y_max))})"
            if init_prompt.box is not None
            else "box=None"
        )
        debug_label = (
            f"run_id={effective_run_id[:12]} frame_index={int(init_prompt.frame_index)} "
            f"{box_desc} {point_desc} {width}x{height}"
        )
        init_debug_image_path = _save_init_prompt_debug_image(
            frame_path=init_frame_path,
            prompt=init_prompt,
            output_path=run_dir / SAM2_INIT_DEBUG_IMAGE_FILENAME,
            label=debug_label,
        )
    except Exception as exc:  # noqa: BLE001 - debug artifact is best-effort
        warnings.append(f"init_debug_image_failed: {exc}")
        init_debug_image_path = None

    _maybe_cleanup_cuda()
    try:
        predictor = get_video_predictor(device=resolved_device)

        masks_by_local_frame, source_by_local_frame = _run_predictor(
            predictor=predictor,
            frames_folder=frames_folder,
            selected_local_frame_index=selected_local_frame_index,
            prompt=init_prompt,
            device=resolved_device,
            warnings=warnings,
        )
    except BaseException as exc:  # noqa: BLE001 - we re-raise with context
        if _is_cuda_oom_exception(exc):
            reset_video_predictor_cache()
            logger.error(
                "SAM 2 predictor ran out of GPU memory on device=%s. "
                "This typically means another process (e.g. the running "
                "'uvicorn' backend AND a CLI invocation) is holding SAM 2 "
                "weights on the same GPU. Stop one of them and retry.",
                resolved_device,
            )
            raise SAM2GPUOutOfMemoryError(
                "SAM 2 ran out of GPU memory. Another process (likely the "
                "backend uvicorn) is already holding SAM 2 on this GPU. "
                "Stop the other SAM 2 consumer (or run this on a different "
                "GPU / CPU) and retry.",
                resolved_device=resolved_device,
            ) from exc
        raise

    masks_output_folder = run_dir / SAM2_MASKS_SUBDIR
    ensure_clean_dir(masks_output_folder)
    frame_records, drawing_records = _build_detections_and_frame_results(
        saved_paths=saved_paths,
        saved_absolute_frame_indices=saved_absolute_frame_indices,
        masks_by_local_frame=masks_by_local_frame,
        source_by_local_frame=source_by_local_frame,
        fps=fps,
        masks_output_folder=masks_output_folder,
        target_obj_id=int(init_prompt.target_object_id),
    )

    annotated_path: Optional[Path] = None
    if render_annotation:
        try:
            annotated_path = _render_annotated_video(
                saved_paths=saved_paths,
                drawing_records=drawing_records,
                output_video_path=run_dir / SAM2_ANNOTATED_FILENAME,
                fps=fps,
                overlay_folder=run_dir / "overlay_frames",
            )
            if annotated_path is None:
                warnings.append("annotated_video_not_written")
        except Exception as exc:  # noqa: BLE001 - rendering is best-effort
            warnings.append(f"annotation_failed: {exc}")
            annotated_path = None

    frames_with_mask = sum(1 for fr in frame_records if fr.has_mask)
    total_frames_processed = len(frame_records)
    detection_rate = (
        frames_with_mask / total_frames_processed if total_frames_processed > 0 else 0.0
    )

    detections_document = SAM2DetectionsDocument(
        run_id=effective_run_id,
        source_video_path=str(resolved_video_path),
        fps=fps,
        frame_count=info["frame_count"],
        width=width,
        height=height,
        target_object_id=int(init_prompt.target_object_id),
        init_prompt=init_prompt,
        frames=frame_records,
    )

    features_document = _build_features_document(
        run_id=effective_run_id,
        source_video_path=resolved_video_path,
        fps=fps,
        frame_count=info["frame_count"],
        image_width=width,
        image_height=height,
        target_object_id=int(init_prompt.target_object_id),
        frame_records=frame_records,
    )

    metadata_document = SAM2RunMeta(
        run_id=effective_run_id,
        source_video_path=str(resolved_video_path),
        pipeline_name=SAM2_PIPELINE_NAME,
        model_name=SAM2_DEFAULT_MODEL_NAME,
        device=resolved_device,
        gpu_name=gpu_name,
        model_checkpoint_path=str(assets.checkpoint_path),
        model_config_path=str(assets.config_path),
        created_at=datetime.now(timezone.utc).isoformat(),
        fps=fps,
        frame_count=info["frame_count"],
        width=width,
        height=height,
        analysis_start_frame_index=window["start_frame_index"],
        analysis_end_frame_index=window["end_frame_index"],
        frame_stride=int(extraction["frame_stride"]),
        total_frames_processed=total_frames_processed,
        frames_with_mask=frames_with_mask,
        detection_rate=detection_rate,
        target_object_id=int(init_prompt.target_object_id),
        init_prompt=init_prompt,
        warnings=warnings,
    )

    detections_path = run_dir / SAM2_DETECTIONS_FILENAME
    features_path = run_dir / SAM2_FEATURES_FILENAME
    metadata_path = run_dir / SAM2_METADATA_FILENAME

    _write_json(detections_path, detections_document)
    _write_json(features_path, features_document)
    _write_json(metadata_path, metadata_document)

    _maybe_cleanup_cuda()

    logger.info(
        "SAM 2 run complete: run_id=%s frames=%s masks=%s (%.2f%%) device=%s",
        effective_run_id,
        total_frames_processed,
        frames_with_mask,
        detection_rate * 100.0,
        resolved_device,
    )

    return SAM2RunArtifacts(
        run_id=effective_run_id,
        run_dir=run_dir,
        detections_path=detections_path,
        features_path=features_path,
        metadata_path=metadata_path,
        annotated_video_path=annotated_path,
        source_video_path=resolved_video_path,
        detections=detections_document,
        metadata=metadata_document,
        features=features_document,
        warnings=warnings,
        init_debug_image_path=init_debug_image_path,
    )


def load_run_artifacts(run_id: str) -> SAM2RunArtifacts:
    """Rehydrate a previously-completed SAM 2 run from disk."""
    run_dir = resolve_run_dir(run_id)
    if not run_dir.is_dir():
        raise SAM2Error(f"SAM 2 run not found: {run_id}")

    detections_path = run_dir / SAM2_DETECTIONS_FILENAME
    features_path = run_dir / SAM2_FEATURES_FILENAME
    metadata_path = run_dir / SAM2_METADATA_FILENAME
    annotated_path = run_dir / SAM2_ANNOTATED_FILENAME
    init_debug_path = run_dir / SAM2_INIT_DEBUG_IMAGE_FILENAME

    if not detections_path.is_file() or not metadata_path.is_file():
        raise SAM2Error(
            f"SAM 2 run {run_id} is missing mandatory JSON artifacts."
        )

    with detections_path.open("r", encoding="utf-8") as handle:
        detections = SAM2DetectionsDocument.model_validate(json.load(handle))
    with metadata_path.open("r", encoding="utf-8") as handle:
        metadata = SAM2RunMeta.model_validate(json.load(handle))
    features: SAM2FeaturesDocument
    if features_path.is_file():
        with features_path.open("r", encoding="utf-8") as handle:
            features = SAM2FeaturesDocument.model_validate(json.load(handle))
    else:
        features = SAM2FeaturesDocument(
            run_id=run_id,
            source_video_path=metadata.source_video_path,
            fps=metadata.fps,
            frame_count=metadata.frame_count,
            width=metadata.width,
            height=metadata.height,
            target_object_id=metadata.target_object_id,
        )

    return SAM2RunArtifacts(
        run_id=run_id,
        run_dir=run_dir,
        detections_path=detections_path,
        features_path=features_path,
        metadata_path=metadata_path,
        annotated_video_path=annotated_path if annotated_path.is_file() else None,
        source_video_path=Path(metadata.source_video_path),
        detections=detections,
        metadata=metadata,
        features=features,
        warnings=list(metadata.warnings),
        init_debug_image_path=init_debug_path if init_debug_path.is_file() else None,
    )
