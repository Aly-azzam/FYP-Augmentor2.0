"""Helper functions for the SAM 2 backend pipeline.

This module is a clean, production-friendly port of the small set of
primitives we kept from ``Old SAM2/`` — specifically:

    * ``video_utils.py`` -> frame extraction and video-window helpers
    * ``sam_runner.py``  -> mask / overlay / bbox / centroid helpers,
                             binary-mask decoding from predictor logits,
                             and frame-to-mp4 writing with ffmpeg fallback

Everything UI-related, the "point picker" flow and the Gradio-era
``track_two_videos_from_selected_points`` orchestration were NOT ported;
they have no role in the automated MediaPipe-initialized backend.

The functions below are deliberately:

    * pure helpers (no DB, no FastAPI, no app state)
    * typed with ``pathlib.Path`` on the public surface
    * tolerant of both tensor and numpy mask inputs
    * JSON-safe (return plain Python ints/floats)
"""

from __future__ import annotations

import logging
import shutil
import subprocess
from pathlib import Path
from typing import Any, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from app.core.sam2_constants import (
    SAM2_ANALYSIS_MODE_FIRST_N_SECONDS,
    SAM2_ANALYSIS_MODE_FULL,
    SAM2_DEFAULT_ANALYSIS_MODE,
    SAM2_DEFAULT_FRAME_STRIDE,
    SAM2_DEFAULT_MAX_SECONDS,
    SAM2_MASK_LOGIT_THRESHOLD,
)


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Video I/O helpers (ported from Old SAM2/video_utils.py)
# ---------------------------------------------------------------------------

def _open_capture(video_path: Path) -> Optional[cv2.VideoCapture]:
    """Return an opened OpenCV capture or ``None`` if the video is unreadable."""
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        return None
    return capture


def get_video_info(video_path: Path) -> dict:
    """Return basic video metadata as a plain dict.

    Returns ``{"error": "..."}`` on failure so callers can surface a
    friendly message without having to try/except around OpenCV.
    """
    capture = _open_capture(video_path)
    if capture is None:
        return {"error": f"Could not open video: {video_path}"}

    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    capture.release()

    duration_seconds = frame_count / fps if fps > 0 else 0.0

    return {
        "frame_count": frame_count,
        "fps": fps,
        "width": width,
        "height": height,
        "duration_seconds": duration_seconds,
    }


def get_analysis_window(
    video_path: Path,
    *,
    analysis_mode: str = SAM2_DEFAULT_ANALYSIS_MODE,
    max_seconds: float = SAM2_DEFAULT_MAX_SECONDS,
) -> dict:
    """Resolve the ``[start_frame, end_frame]`` window to process.

    * ``analysis_mode="full"`` processes every frame.
    * ``analysis_mode="first_n_seconds"`` clips the end frame to
      ``max_seconds`` seconds of footage.
    """
    info = get_video_info(video_path)
    if "error" in info:
        return info

    frame_count = info["frame_count"]
    fps = info["fps"]

    if frame_count <= 0:
        return {"error": "Video has no frames or reports an invalid frame count."}

    if analysis_mode == SAM2_ANALYSIS_MODE_FULL:
        end_frame_index = frame_count - 1
    elif analysis_mode == SAM2_ANALYSIS_MODE_FIRST_N_SECONDS:
        if fps <= 0:
            return {"error": "Invalid FPS; cannot clip by seconds."}
        end_frame_index = int(max_seconds * fps) - 1
        end_frame_index = min(end_frame_index, frame_count - 1)
    else:
        return {"error": f"Unknown analysis_mode: {analysis_mode!r}"}

    start_frame_index = 0
    start_time_seconds = 0.0
    end_time_seconds = end_frame_index / fps if fps > 0 else 0.0

    return {
        "start_frame_index": start_frame_index,
        "end_frame_index": end_frame_index,
        "frame_count_to_process": end_frame_index - start_frame_index + 1,
        "start_time_seconds": start_time_seconds,
        "end_time_seconds": end_time_seconds,
    }


def ensure_clean_dir(folder_path: Path) -> Path:
    """Recreate ``folder_path`` from scratch (used for frame / overlay folders)."""
    folder_path = Path(folder_path)
    if folder_path.exists():
        shutil.rmtree(folder_path)
    folder_path.mkdir(parents=True, exist_ok=True)
    return folder_path


def extract_frame_range_to_folder(
    video_path: Path,
    output_folder: Path,
    *,
    start_frame_index: int,
    end_frame_index: int,
    frame_stride: int = SAM2_DEFAULT_FRAME_STRIDE,
) -> dict:
    """Decode the requested frame range to ``output_folder`` as JPEGs.

    Returns a dict with the saved frame paths and their absolute frame
    indices, or ``{"error": "..."}`` on failure.
    """
    video_path = Path(video_path)
    output_folder = Path(output_folder)

    if frame_stride is None or int(frame_stride) <= 0:
        return {"error": "frame_stride must be >= 1"}
    frame_stride = int(frame_stride)

    capture = _open_capture(video_path)
    if capture is None:
        return {"error": f"Could not open video: {video_path}"}

    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if total_frames <= 0:
        capture.release()
        return {"error": "Video has no frames or reports an invalid frame count."}

    start_frame_index = max(0, int(start_frame_index))
    end_frame_index = min(int(end_frame_index), total_frames - 1)

    if start_frame_index > end_frame_index:
        capture.release()
        return {"error": "Invalid frame range."}

    ensure_clean_dir(output_folder)

    saved_paths: List[Path] = []
    saved_absolute_frame_indices: List[int] = []

    capture.set(cv2.CAP_PROP_POS_FRAMES, start_frame_index)
    current_frame_index = start_frame_index
    try:
        while current_frame_index <= end_frame_index:
            success, frame = capture.read()
            if not success or frame is None:
                break

            if (current_frame_index - start_frame_index) % frame_stride == 0:
                frame_name = f"{current_frame_index:06d}.jpg"
                frame_path = output_folder / frame_name
                cv2.imwrite(str(frame_path), frame)
                saved_paths.append(frame_path)
                saved_absolute_frame_indices.append(current_frame_index)

            current_frame_index += 1
    finally:
        capture.release()

    return {
        "output_folder": output_folder,
        "saved_frame_count": len(saved_paths),
        "saved_paths": saved_paths,
        "saved_absolute_frame_indices": saved_absolute_frame_indices,
        "start_frame_index": saved_absolute_frame_indices[0] if saved_absolute_frame_indices else None,
        "end_frame_index": saved_absolute_frame_indices[-1] if saved_absolute_frame_indices else None,
        "frame_stride": frame_stride,
    }


# ---------------------------------------------------------------------------
# Mask geometry helpers (ported from Old SAM2/sam_runner.py)
# ---------------------------------------------------------------------------

def mask_to_overlay(image_bgr: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Return a copy of ``image_bgr`` with the masked region tinted green.

    The tint is an opaque paint-over, matching the original demo. If a
    blended overlay is wanted later, replace this with ``addWeighted``.
    """
    overlay = image_bgr.copy()
    overlay[mask.astype(bool)] = [0, 255, 0]
    return overlay


def mask_area(mask: np.ndarray) -> int:
    """Number of foreground pixels in the binary mask."""
    return int(np.asarray(mask).astype(bool).sum())


def mask_centroid(mask: np.ndarray) -> Optional[Tuple[int, int]]:
    """Return the ``(x, y)`` centroid in pixel coordinates, or ``None``."""
    ys, xs = np.asarray(mask).astype(bool).nonzero()
    if xs.size == 0 or ys.size == 0:
        return None
    return int(xs.mean()), int(ys.mean())


def mask_bbox(mask: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
    """Tight axis-aligned bbox ``(x_min, y_min, x_max, y_max)`` or ``None``."""
    ys, xs = np.asarray(mask).astype(bool).nonzero()
    if xs.size == 0 or ys.size == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


def save_binary_mask_image(mask: Optional[np.ndarray], output_path: Path) -> Optional[Path]:
    """Persist a binary mask as an 8-bit PNG (0 / 255) and return its path."""
    if mask is None:
        return None
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    mask_image = (np.asarray(mask).astype(np.uint8) * 255)
    cv2.imwrite(str(output_path), mask_image)
    return output_path.resolve()


# ---------------------------------------------------------------------------
# Predictor output decoding
# ---------------------------------------------------------------------------

def select_object_index(
    out_obj_ids: Optional[Sequence[int]],
    target_obj_id: int,
) -> Optional[int]:
    """Return the index within ``out_obj_ids`` that matches ``target_obj_id``.

    Falls back to index ``0`` when the target id is not present (matching
    the defensive behaviour of the old demo), or ``None`` when the
    predictor returned no objects at all.
    """
    if out_obj_ids is None:
        return None
    if len(out_obj_ids) == 0:
        return None
    for i, current_obj_id in enumerate(out_obj_ids):
        if int(current_obj_id) == int(target_obj_id):
            return i
    return 0


def logits_to_binary_mask(
    mask_logits: Any,
    *,
    threshold: float = SAM2_MASK_LOGIT_THRESHOLD,
) -> np.ndarray:
    """Decode SAM 2 predictor logits into a ``uint8`` binary mask.

    Accepts either a torch tensor or a numpy array; returns numpy (the
    rest of the pipeline works in numpy/cv2 space).
    """
    try:
        import torch  # type: ignore
    except ImportError:  # pragma: no cover - torch is required at runtime
        torch = None  # type: ignore

    if torch is not None and torch.is_tensor(mask_logits):
        tensor = mask_logits.detach()
        tensor = torch.squeeze(tensor)
        binary = (tensor > float(threshold)).to(torch.uint8)
        return binary.to("cpu", non_blocking=True).numpy()

    array = np.asarray(mask_logits)
    array = np.squeeze(array)
    return (array > float(threshold)).astype(np.uint8)


def get_binary_mask_for_object(
    out_obj_ids: Optional[Sequence[int]],
    out_mask_logits: Any,
    *,
    target_obj_id: int,
    threshold: float = SAM2_MASK_LOGIT_THRESHOLD,
) -> Optional[np.ndarray]:
    """Extract the binary mask for ``target_obj_id`` from predictor output."""
    if out_mask_logits is None:
        return None
    object_index = select_object_index(out_obj_ids, target_obj_id)
    if object_index is None:
        return None
    try:
        mask_tensor = out_mask_logits[object_index]
    except (IndexError, TypeError):
        return None
    return logits_to_binary_mask(mask_tensor, threshold=threshold)


# ---------------------------------------------------------------------------
# Video writing (ported, trimmed)
# ---------------------------------------------------------------------------

def _write_temp_video_from_frames(
    frame_paths: Sequence[Path],
    output_video_path: Path,
    *,
    fps: float,
) -> bool:
    """Write ``frame_paths`` to an mp4v mp4 (intermediate, no web flags)."""
    if len(frame_paths) == 0:
        return False

    first_frame = cv2.imread(str(frame_paths[0]))
    if first_frame is None:
        return False

    height, width = first_frame.shape[:2]
    output_video_path.parent.mkdir(parents=True, exist_ok=True)

    writer = cv2.VideoWriter(
        str(output_video_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        float(fps) if fps > 0 else 30.0,
        (width, height),
    )
    try:
        for frame_path in frame_paths:
            frame = cv2.imread(str(frame_path))
            if frame is None:
                continue
            writer.write(frame)
    finally:
        writer.release()
    return True


def _convert_video_to_web_mp4(input_video_path: Path, output_video_path: Path) -> bool:
    """Re-encode ``input_video_path`` to a browser-friendly H.264 mp4.

    Falls back gracefully: if ``imageio_ffmpeg`` is not installed or the
    call fails, the caller keeps the mp4v intermediate.
    """
    try:
        import imageio_ffmpeg  # type: ignore
    except ImportError:
        logger.info("imageio_ffmpeg not installed; skipping web mp4 re-encode.")
        return False

    try:
        ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
    except Exception as exc:  # noqa: BLE001 - defensive, library behaviour varies
        logger.warning("Could not locate ffmpeg: %s", exc)
        return False

    command = [
        ffmpeg_exe,
        "-y",
        "-i", str(input_video_path),
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
        "-an",
        str(output_video_path),
    ]

    result = subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    return result.returncode == 0


def write_video_from_frames(
    frame_paths: Sequence[Path],
    output_video_path: Path,
    *,
    fps: float,
) -> Optional[Path]:
    """Write an mp4 from still frames; prefer a browser-playable H.264 output.

    Returns the path of the final video (either the H.264 web version or
    the mp4v fallback), or ``None`` if nothing could be written.
    """
    output_video_path = Path(output_video_path)
    temp_output_video_path = output_video_path.with_name(
        output_video_path.stem + "_temp" + output_video_path.suffix
    )
    web_output_video_path = output_video_path.with_name(
        output_video_path.stem + "_web" + output_video_path.suffix
    )

    if not _write_temp_video_from_frames(
        frame_paths=frame_paths,
        output_video_path=temp_output_video_path,
        fps=fps,
    ):
        return None

    if _convert_video_to_web_mp4(temp_output_video_path, web_output_video_path):
        try:
            temp_output_video_path.unlink()
        except OSError:
            pass
        # Normalize the final filename to the one the caller asked for.
        try:
            if output_video_path.exists():
                output_video_path.unlink()
            web_output_video_path.rename(output_video_path)
            return output_video_path.resolve()
        except OSError:
            return web_output_video_path.resolve()

    # Fallback path: return the mp4v intermediate under the requested name.
    try:
        if output_video_path.exists():
            output_video_path.unlink()
        temp_output_video_path.rename(output_video_path)
        return output_video_path.resolve()
    except OSError:
        return temp_output_video_path.resolve()


# ---------------------------------------------------------------------------
# MediaPipe -> SAM 2 initializer bridge (pure helpers)
# ---------------------------------------------------------------------------
#
# SAM 2 is initialized automatically from MediaPipe output. The pure
# coordinate + landmark-selection helpers live here so they can be reused
# from tests / CLI scripts without pulling in the whole service layer
# (which also handles file I/O and orchestration).
#
# Design note: we use the MediaPipe index fingertip as the default proxy
# for the "action zone" because it is the single most stable anchor
# across both drawing and cutting tasks. SAM 2 does not need a perfect
# tool-specific click — it only needs a point that lives inside, or
# immediately next to, the object we want to segment.

#: Names must match ``MediaPipeFrameFeatures.wrist_relative_landmarks``
#: keys (which come from ``HAND_LANDMARK_NAMES`` in the MediaPipe utils).
_LANDMARK_NAME_INDEX_TIP: str = "index_finger_tip"
_LANDMARK_NAME_MIDDLE_TIP: str = "middle_finger_tip"
_LANDMARK_NAME_THUMB_TIP: str = "thumb_tip"

#: Public shorthands accepted as ``preferred_landmark`` arguments.
LANDMARK_ALIAS_INDEX_TIP: str = "index_tip"
LANDMARK_ALIAS_MIDDLE_TIP: str = "middle_tip"
LANDMARK_ALIAS_THUMB_TIP: str = "thumb_tip"
LANDMARK_ALIAS_HAND_CENTER: str = "hand_center"
LANDMARK_ALIAS_WRIST: str = "wrist"

#: Fallback chain used when the preferred landmark is unavailable. The
#: chain is ordered by how reliably MediaPipe produces that signal across
#: realistic FYP footage (tip > tip > tip > palm center > wrist anchor).
DEFAULT_LANDMARK_FALLBACK_ORDER: Tuple[str, ...] = (
    LANDMARK_ALIAS_INDEX_TIP,
    LANDMARK_ALIAS_MIDDLE_TIP,
    LANDMARK_ALIAS_THUMB_TIP,
    LANDMARK_ALIAS_HAND_CENTER,
    LANDMARK_ALIAS_WRIST,
)

#: Map public aliases to the underlying ``wrist_relative_landmarks`` key.
_ALIAS_TO_RELATIVE_LANDMARK: dict[str, str] = {
    LANDMARK_ALIAS_INDEX_TIP: _LANDMARK_NAME_INDEX_TIP,
    LANDMARK_ALIAS_MIDDLE_TIP: _LANDMARK_NAME_MIDDLE_TIP,
    LANDMARK_ALIAS_THUMB_TIP: _LANDMARK_NAME_THUMB_TIP,
}


def normalized_to_pixel_coords(
    nx: float,
    ny: float,
    image_width: int,
    image_height: int,
) -> Tuple[float, float]:
    """Denormalize MediaPipe's ``[0, 1]`` coordinates to pixel space.

    MediaPipe reports ``x`` in ``[0, 1]`` relative to image width and
    ``y`` in ``[0, 1]`` relative to image height. SAM 2 expects raw
    pixel coordinates, so we multiply through and then clamp to the
    image bounds to defend against landmarks that drift slightly
    outside the frame.
    """
    if image_width <= 0 or image_height <= 0:
        raise ValueError(
            f"image_width/image_height must be positive, got "
            f"({image_width}, {image_height})."
        )

    px = float(nx) * float(image_width)
    py = float(ny) * float(image_height)

    px = max(0.0, min(float(image_width - 1), px))
    py = max(0.0, min(float(image_height - 1), py))
    return px, py


def _frame_has_detection(frame: Any) -> bool:
    """Return ``True`` if the frame's ``has_detection`` is truthy."""
    has_detection = getattr(frame, "has_detection", None)
    if has_detection is None and isinstance(frame, dict):
        has_detection = frame.get("has_detection")
    return bool(has_detection)


def _frame_attr(frame: Any, name: str) -> Any:
    """Dict- and model-friendly attribute lookup (returns ``None`` if missing)."""
    value = getattr(frame, name, None)
    if value is None and isinstance(frame, dict):
        value = frame.get(name)
    return value


def _absolute_landmark_xy(frame: Any, alias: str) -> Optional[Tuple[float, float]]:
    """Return the absolute normalized ``(x, y)`` for one named landmark.

    ``hand_center`` and ``wrist`` are already stored absolute; fingertip
    aliases have to be reconstructed as ``wrist + wrist_relative`` because
    ``features.json`` only persists the wrist-relative offsets (see
    ``feature_service.build_features_document``).

    Returns ``None`` when the requested landmark is unavailable on that
    frame (for example when the detector dropped it).
    """
    if alias == LANDMARK_ALIAS_HAND_CENTER:
        candidate = _frame_attr(frame, "hand_center")
        if not candidate or len(candidate) < 2:
            return None
        return float(candidate[0]), float(candidate[1])

    if alias == LANDMARK_ALIAS_WRIST:
        candidate = _frame_attr(frame, "wrist")
        if not candidate or len(candidate) < 2:
            return None
        return float(candidate[0]), float(candidate[1])

    relative_key = _ALIAS_TO_RELATIVE_LANDMARK.get(alias)
    if relative_key is None:
        return None

    wrist = _frame_attr(frame, "wrist")
    if not wrist or len(wrist) < 2:
        return None

    relative_dict = _frame_attr(frame, "wrist_relative_landmarks")
    if not relative_dict:
        return None
    # ``wrist_relative_landmarks`` is a plain dict on the pydantic model too.
    offset = relative_dict.get(relative_key) if isinstance(relative_dict, dict) else None
    if not offset or len(offset) < 2:
        return None

    return float(wrist[0]) + float(offset[0]), float(wrist[1]) + float(offset[1])


def select_prompt_point_from_mediapipe_frame(
    frame: Any,
    *,
    preferred_landmark: str = LANDMARK_ALIAS_INDEX_TIP,
    fallback_order: Sequence[str] = DEFAULT_LANDMARK_FALLBACK_ORDER,
) -> Optional[Tuple[Tuple[float, float], str]]:
    """Pick a single ``(x, y)`` anchor (still normalized) from one MediaPipe frame.

    The lookup tries ``preferred_landmark`` first, then each entry of
    ``fallback_order`` that isn't the preferred one. Returns the first
    landmark that resolves to a valid 2D coordinate, along with the
    alias name used (so callers can record "index_tip" vs
    "middle_tip" in their debug info).

    Returns ``None`` when the frame has no detection OR when none of the
    candidate landmarks are available (e.g. MediaPipe dropped the
    ``wrist_relative_landmarks`` dict for that frame).
    """
    if not _frame_has_detection(frame):
        return None

    ordered_candidates: List[str] = []
    if preferred_landmark not in ordered_candidates:
        ordered_candidates.append(preferred_landmark)
    for alias in fallback_order:
        if alias not in ordered_candidates:
            ordered_candidates.append(alias)

    for alias in ordered_candidates:
        xy = _absolute_landmark_xy(frame, alias)
        if xy is None:
            continue
        # Reject ([0, 0]) sentinel values coming from a dropped landmark
        # vector — MediaPipe often reports a genuine landmark somewhere
        # strictly inside (0, 1) so this is a safe defensive check.
        if xy == (0.0, 0.0):
            continue
        return xy, alias

    return None


def iter_mask_records(masks_by_frame: dict[int, np.ndarray]) -> Iterable[Tuple[int, np.ndarray]]:
    """Iterate ``(frame_index, mask)`` pairs sorted by frame index."""
    for frame_index in sorted(masks_by_frame.keys()):
        yield frame_index, masks_by_frame[frame_index]
