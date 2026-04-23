"""Shared constants and default configuration for the SAM 2 pipeline.

These values are the single source of truth for the SAM 2 backend:
schemas, utils, and services all import from this module rather than
redefining paths / thresholds locally.

Model assets live under ``backend/models/sam2/`` so that large ``*.pt``
checkpoints stay out of the Python package and can be managed separately
(mirroring how ``hand_landmarker.task`` is stored for MediaPipe).

None of these constants imply live inference yet; the service layer is
free to read them lazily when a predictor is actually needed.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from app.core.config import settings


# ---------------------------------------------------------------------------
# Filesystem layout
# ---------------------------------------------------------------------------

#: Root folder for transient SAM 2 run outputs. Mirrors the MediaPipe layout
#: at ``storage/mediapipe/runs/<run_id>/`` so tooling is interchangeable.
SAM2_RUNS_SUBDIR: str = "sam2/runs"

#: Folder under STORAGE_ROOT where stable expert SAM 2 references live.
#: Mirrors ``storage/expert/mediapipe/<expert_code>/`` from the MediaPipe
#: expert flow. Actual subfolder is created by the service per expert.
SAM2_EXPERT_SUBDIR: str = "expert/sam2"

#: Root folder for model weights and configs. Checkpoints are intentionally
#: NOT bundled in the repository — operators must drop them here manually.
SAM2_MODELS_ROOT: Path = Path(settings.STORAGE_ROOT).parent / "models" / "sam2"

#: Legacy asset layout used by the old SAM2 demo.
#:
#: Old code referenced these relative locations:
#:   checkpoints/sam2_hiera_tiny.pt
#:   configs/sam2/sam2_hiera_t.yaml
#:
#: AugMentor backend standardizes both files under ``backend/models/sam2/``.
SAM2_LEGACY_DEMO_CHECKPOINT_REL: str = "checkpoints/sam2_hiera_tiny.pt"
SAM2_LEGACY_DEMO_CONFIG_REL: str = "configs/sam2/sam2_hiera_t.yaml"

#: Default SAM 2 checkpoint (tiny variant, same one used by the old demo).
SAM2_DEFAULT_CHECKPOINT: Path = SAM2_MODELS_ROOT / "sam2_hiera_tiny.pt"

#: Default SAM 2 model config file (Hiera tiny).
SAM2_DEFAULT_MODEL_CFG_FILENAME: str = "sam2_hiera_t.yaml"
SAM2_DEFAULT_MODEL_CFG_PATH: Path = SAM2_MODELS_ROOT / SAM2_DEFAULT_MODEL_CFG_FILENAME

#: Backward-compatible alias used by existing service signatures.
SAM2_DEFAULT_MODEL_CFG: str = str(SAM2_DEFAULT_MODEL_CFG_PATH)


@dataclass(frozen=True)
class SAM2AssetPaths:
    """Canonical SAM2 asset paths resolved by backend constants."""

    checkpoint_path: Path
    config_path: Path
    legacy_checkpoint_rel: str
    legacy_config_rel: str


def get_sam2_asset_paths() -> SAM2AssetPaths:
    """Return canonical SAM2 checkpoint/config paths for this backend."""
    return SAM2AssetPaths(
        checkpoint_path=SAM2_DEFAULT_CHECKPOINT.expanduser().resolve(),
        config_path=SAM2_DEFAULT_MODEL_CFG_PATH.expanduser().resolve(),
        legacy_checkpoint_rel=SAM2_LEGACY_DEMO_CHECKPOINT_REL,
        legacy_config_rel=SAM2_LEGACY_DEMO_CONFIG_REL,
    )

#: File names for the JSON artifacts produced by a single SAM 2 run.
#:
#: Two tiers of artifacts live side by side inside each run folder:
#:
#:   * ``raw.json`` + ``summary.json`` + ``metadata.json`` — the stable,
#:     downstream-facing contract mirrored into ``storage/expert/sam2/``
#:     for expert references and consumed by the compare / inspection
#:     flows. See ``schemas/sam2/sam2_contract_schema.py``.
#:
#:   * ``detections.json`` + ``features.json`` — richer internal debug
#:     artifacts (per-frame mask paths, temporal source, working-region
#:     aggregation details, etc.). They are intentionally NOT part of
#:     the stable expert reference.
SAM2_RAW_FILENAME: str = "raw.json"
SAM2_SUMMARY_FILENAME: str = "summary.json"
SAM2_METADATA_FILENAME: str = "metadata.json"
SAM2_ANNOTATED_FILENAME: str = "annotated.mp4"
SAM2_SOURCE_FILENAME: str = "source.mp4"

SAM2_DETECTIONS_FILENAME: str = "detections.json"
SAM2_FEATURES_FILENAME: str = "features.json"

SAM2_MASKS_SUBDIR: str = "masks"
SAM2_FRAMES_SUBDIR: str = "frames"


# ---------------------------------------------------------------------------
# Inference defaults
# ---------------------------------------------------------------------------

#: Numeric object id assigned to the single target track. SAM 2 supports
#: multiple objects per video; in AugMentor we track exactly one hand.
SAM2_DEFAULT_TARGET_OBJECT_ID: int = 1

#: Threshold applied to predictor logits to produce a binary mask.
SAM2_MASK_LOGIT_THRESHOLD: float = 0.0

#: Default video analysis window. "full" means "process the whole video";
#: "first_n_seconds" clips to ``SAM2_DEFAULT_MAX_SECONDS``.
SAM2_ANALYSIS_MODE_FULL: str = "full"
SAM2_ANALYSIS_MODE_FIRST_N_SECONDS: str = "first_n_seconds"
SAM2_DEFAULT_ANALYSIS_MODE: str = SAM2_ANALYSIS_MODE_FULL
SAM2_DEFAULT_MAX_SECONDS: float = 15.0

#: Sample every Nth frame from the analysis window.
#:
#: We intentionally default to 3 for runtime performance and to mirror the
#: previous SAM 2 demo/test profile.
SAM2_DEFAULT_FRAME_STRIDE: int = 3

#: Types of initializer prompts we support in the backend. The pipeline
#: only consumes prompts produced by MediaPipe — the human-in-the-loop
#: Gradio "point picker" flow is intentionally not reused.
SAM2_PROMPT_TYPE_POINT: str = "point"
SAM2_PROMPT_TYPE_BOX: str = "box"

#: Number of consecutive valid MediaPipe detection frames we average over when
#: building the SAM 2 initialization prompt. Using several frames instead of
#: a single one smooths out noisy landmarks and gives a more reliable anchor.
SAM2_PROMPT_AVERAGE_FRAME_COUNT: int = 5

#: Size of the candidate pool we scan when selecting a stable prompt window.
#: We scan the first N valid MediaPipe frames, score each candidate, and
#: pick the best ``SAM2_PROMPT_AVERAGE_FRAME_COUNT``-wide window to average.
#: Increasing this makes the initializer more robust to a noisy video start
#: (hand entering frame, motion blur, partial clipping) at a tiny CPU cost.
SAM2_PROMPT_CANDIDATE_POOL_SIZE: int = 60

#: Minimum bbox area as a fraction of image area for a candidate prompt frame
#: to be accepted. Frames whose hand bbox is absurdly tiny almost always mean
#: the hand is barely in view and the resulting prompt would be useless.
SAM2_PROMPT_MIN_BBOX_AREA_RATIO: float = 0.005

#: Maximum bbox area as a fraction of image area for a candidate prompt frame
#: to be accepted. Frames whose hand bbox covers most of the frame almost
#: always mean MediaPipe's bbox drifted off (hand + forearm + background), and
#: initializing SAM 2 from that box makes SAM 2 latch onto the background.
SAM2_PROMPT_MAX_BBOX_AREA_RATIO: float = 0.55

#: Normalized margin from each image edge. Candidate frames whose hand bbox
#: touches (or crosses) the edge within this margin are penalized because
#: the hand is likely clipped and the bbox is not trustworthy.
SAM2_PROMPT_BORDER_MARGIN_NORM: float = 0.01

#: Normalized padding added around the MediaPipe hand bbox when we convert it
#: into the SAM 2 initialization box. Expressed as a fraction of the bbox
#: width/height (e.g. 0.10 = 10% padding on each side). Kept moderate so the
#: SAM 2 box remains hand-focused instead of capturing notebook / background.
SAM2_PROMPT_BOX_PADDING_RATIO: float = 0.15

#: When the anchor point falls outside the final padded bbox (can happen with
#: a jumpy fingertip vs. a laggy hand bbox) we re-anchor it to the box center.
#: This guarantees SAM 2 always sees ``point inside box``.
SAM2_PROMPT_POINT_INSIDE_BOX_MARGIN_PX: float = 2.0

#: Pixel offset used to project a pen-tip proxy ahead of the grip center
#: in the index-finger direction.
SAM2_PEN_TIP_OFFSET_PX: float = 28.0

#: Side length in pixels for the optional local box centered on the
#: estimated pen-tip prompt.
SAM2_PEN_TIP_BOX_SIZE_PX: float = 72.0

#: Filename for the init-prompt debug image saved inside each SAM 2 run
#: folder. The image shows the selected initialization frame with the point
#: and/or box overlay so we can verify the prompt visually.
SAM2_INIT_DEBUG_IMAGE_FILENAME: str = "init_debug.png"
SAM2_SUPPORTED_PROMPT_TYPES: tuple[str, ...] = (
    SAM2_PROMPT_TYPE_POINT,
    SAM2_PROMPT_TYPE_BOX,
)

#: Marker used in logs / metadata to document that a prompt came from the
#: MediaPipe output (not from an interactive UI click).
SAM2_PROMPT_SOURCE_MEDIAPIPE: str = "mediapipe"
SAM2_PROMPT_SOURCE_MANUAL: str = "manual"

#: Pipeline name stamped onto JSON artifacts so downstream consumers can
#: distinguish SAM 2 outputs from MediaPipe ones.
SAM2_PIPELINE_NAME: str = "sam2_video_segmentation"
SAM2_DEFAULT_MODEL_NAME: str = "sam2_hiera_tiny"


# ---------------------------------------------------------------------------
# Working-region defaults (used by feature_service / future integration)
# ---------------------------------------------------------------------------

#: Padding (in pixels) added around the aggregated hand-mask bbox to
#: produce the "working region" rectangle that downstream mistake logic
#: can check against. Keep conservative by default.
SAM2_WORKING_REGION_PADDING_PX: int = 20

#: Quantile used when aggregating per-frame mask bboxes into the stable
#: working region. 0.95 means "95th percentile of frame bboxes", which is
#: robust to a handful of noisy outlier frames.
SAM2_WORKING_REGION_BBOX_QUANTILE: float = 0.95
