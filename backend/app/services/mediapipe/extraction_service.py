"""MediaPipe Hands extraction service.

This service walks a video frame by frame with OpenCV, runs MediaPipe
Hands on each frame, picks a single hand per frame (prefer right hand,
otherwise the first detected hand), and writes two JSON documents into a
dedicated run folder under ``backend/storage/mediapipe/runs/<run_id>/``:

    detections.json   -> raw per-frame extraction (MediaPipeDetectionsDocument)
    metadata.json     -> run summary                (MediaPipeRunMeta)

Only MediaPipe Hands is used. There is no Pose fallback. DTW, SAM2 and
optical flow are intentionally out of scope for this step.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, List, Optional
from uuid import uuid4

import cv2
import mediapipe as mp

from app.core.config import settings
from app.schemas.mediapipe.mediapipe_schema import (
    MediaPipeDetectionsDocument,
    MediaPipeFrameRaw,
    MediaPipeLandmarkPoint,
    MediaPipeRunMeta,
    MediaPipeSelectedHandRaw,
)
from app.utils.mediapipe.mediapipe_utils import (
    HAND_LANDMARK_NAMES,
    bgr_to_rgb,
    compute_hand_center,
    landmark_to_list,
    normalize_handedness_label,
)


logger = logging.getLogger(__name__)

SELECTED_HAND_POLICY_PREFER_RIGHT = "prefer_right_else_first"

_WRIST_INDEX = 0
_THUMB_TIP_INDEX = 4
_INDEX_TIP_INDEX = 8
_MIDDLE_TIP_INDEX = 12


class MediaPipeExtractionError(RuntimeError):
    """Raised when MediaPipe extraction cannot complete."""


@dataclass
class MediaPipeExtractionResult:
    """Paths and documents produced by a single extraction run."""

    run_id: str
    run_dir: Path
    detections_path: Path
    metadata_path: Path
    detections: MediaPipeDetectionsDocument
    metadata: MediaPipeRunMeta


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _default_runs_root() -> Path:
    return Path(settings.STORAGE_ROOT) / "mediapipe" / "runs"


def _initialize_hands_detector(
    max_num_hands: int,
    min_detection_confidence: float,
    min_tracking_confidence: float,
) -> Any:
    """Build a MediaPipe ``HandLandmarker`` via the Tasks API.

    The existing AugMentor perception pipeline already relies on the Tasks
    API (``mp.tasks.vision.HandLandmarker``) and ships a
    ``hand_landmarker.task`` model under ``backend/models/``. We reuse both
    to stay consistent with the rest of the project.
    """
    model_path = Path(settings.HAND_LANDMARKER_MODEL_PATH)
    if not model_path.is_file():
        raise MediaPipeExtractionError(
            f"Hand landmarker model file not found: {model_path}"
        )

    base_options = mp.tasks.BaseOptions(model_asset_path=str(model_path))
    options = mp.tasks.vision.HandLandmarkerOptions(
        base_options=base_options,
        running_mode=mp.tasks.vision.RunningMode.IMAGE,
        num_hands=max_num_hands,
        min_hand_detection_confidence=min_detection_confidence,
        min_hand_presence_confidence=min_detection_confidence,
        min_tracking_confidence=min_tracking_confidence,
    )
    return mp.tasks.vision.HandLandmarker.create_from_options(options)


def _select_hand_index(handedness_per_hand: Optional[list]) -> int:
    """Prefer a right hand when present, otherwise take the first detection.

    ``handedness_per_hand`` is the Tasks API ``results.handedness`` list:
    ``list[list[Category]]``. We look at the top-ranked category for each
    hand and return the first index whose normalized label is ``"Right"``;
    if none is found, we default to index ``0``.
    """
    if not handedness_per_hand:
        return 0
    for hand_index, categories in enumerate(handedness_per_hand):
        if not categories:
            continue
        top_category = categories[0]
        raw_label = getattr(top_category, "category_name", None)
        if raw_label is None:
            raw_label = getattr(top_category, "display_name", None)
        label = normalize_handedness_label(raw_label)
        if label == "Right":
            return hand_index
    return 0


def _build_selected_hand(
    hand_landmark_list: Any,
    handedness_categories: Any,
) -> MediaPipeSelectedHandRaw:
    """Build a ``MediaPipeSelectedHandRaw`` from a single Tasks API detection.

    ``hand_landmark_list`` is the list of 21 ``NormalizedLandmark`` objects
    (``results.hand_landmarks[hand_index]``) and ``handedness_categories``
    is the matching list of ``Category`` objects
    (``results.handedness[hand_index]``).
    """
    if not hand_landmark_list or len(hand_landmark_list) != 21:
        raise MediaPipeExtractionError(
            "MediaPipe Hands must return exactly 21 landmarks per detected hand."
        )

    landmark_vectors: List[List[float]] = [landmark_to_list(lm) for lm in hand_landmark_list]

    landmark_points: List[MediaPipeLandmarkPoint] = [
        MediaPipeLandmarkPoint(
            index=index,
            name=HAND_LANDMARK_NAMES[index],
            x=coords[0],
            y=coords[1],
            z=coords[2],
        )
        for index, coords in enumerate(landmark_vectors)
    ]

    if handedness_categories:
        top_category = handedness_categories[0]
        raw_label = getattr(top_category, "category_name", None) or getattr(
            top_category, "display_name", None
        )
        raw_score = float(getattr(top_category, "score", 0.0))
    else:
        raw_label = None
        raw_score = 0.0
    normalized_label = normalize_handedness_label(raw_label)

    # The Tasks API doesn't expose a separate per-hand detection confidence
    # distinct from the handedness score, so we use the classification score
    # as a reasonable proxy to keep the schema self-contained.
    detection_confidence = max(0.0, min(1.0, raw_score))

    hand_center_xy = compute_hand_center(landmark_vectors)

    return MediaPipeSelectedHandRaw(
        handedness=normalized_label,
        handedness_score=max(0.0, min(1.0, raw_score)),
        detection_confidence=detection_confidence,
        wrist=landmark_vectors[_WRIST_INDEX],
        index_tip=landmark_vectors[_INDEX_TIP_INDEX],
        thumb_tip=landmark_vectors[_THUMB_TIP_INDEX],
        middle_tip=landmark_vectors[_MIDDLE_TIP_INDEX],
        hand_center=hand_center_xy,
        landmarks=landmark_points,
    )


def _write_json(path: Path, document) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # ``model_dump`` emits plain Python types, which ``json.dump`` serializes
    # losslessly. We avoid ``model_dump_json`` so we can enforce pretty output.
    payload = document.model_dump(mode="json")
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_extraction(
    video_path: str | Path,
    *,
    run_id: Optional[str] = None,
    runs_root: Optional[Path] = None,
    max_num_hands: int = 2,
    min_detection_confidence: float = 0.5,
    min_tracking_confidence: float = 0.5,
    selected_hand_policy: str = SELECTED_HAND_POLICY_PREFER_RIGHT,
) -> MediaPipeExtractionResult:
    """Run MediaPipe Hands extraction on a single video.

    Produces ``detections.json`` and ``metadata.json`` inside
    ``backend/storage/mediapipe/runs/<run_id>/``.
    """
    source_path = Path(video_path).resolve()
    if not source_path.is_file():
        raise MediaPipeExtractionError(f"Video file not found: {source_path}")

    capture = cv2.VideoCapture(str(source_path))
    if not capture.isOpened():
        raise MediaPipeExtractionError(f"OpenCV could not open video: {source_path}")

    fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)

    if width <= 0 or height <= 0:
        capture.release()
        raise MediaPipeExtractionError(
            f"Video has invalid dimensions ({width}x{height}): {source_path}"
        )

    effective_run_id = run_id or uuid4().hex
    runs_root_path = Path(runs_root).resolve() if runs_root else _default_runs_root()
    run_dir = runs_root_path / effective_run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    detections_frames: List[MediaPipeFrameRaw] = []
    frames_with_detection = 0
    right_hand_selected_count = 0
    left_hand_selected_count = 0
    total_frames_processed = 0

    detector = _initialize_hands_detector(
        max_num_hands=max_num_hands,
        min_detection_confidence=min_detection_confidence,
        min_tracking_confidence=min_tracking_confidence,
    )

    try:
        frame_index = 0
        while True:
            success, bgr_frame = capture.read()
            if not success or bgr_frame is None:
                break

            timestamp_sec = frame_index / fps if fps > 0 else 0.0

            rgb_frame = bgr_to_rgb(bgr_frame)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            results = detector.detect(mp_image)

            hand_landmarks_per_hand = getattr(results, "hand_landmarks", None) or []
            handedness_per_hand = getattr(results, "handedness", None) or []
            num_hands_detected = len(hand_landmarks_per_hand)

            selected_hand: Optional[MediaPipeSelectedHandRaw] = None
            if num_hands_detected > 0:
                chosen_index = _select_hand_index(handedness_per_hand)
                chosen_index = min(chosen_index, num_hands_detected - 1)
                categories = (
                    handedness_per_hand[chosen_index]
                    if chosen_index < len(handedness_per_hand)
                    else []
                )
                try:
                    selected_hand = _build_selected_hand(
                        hand_landmarks_per_hand[chosen_index],
                        categories,
                    )
                except MediaPipeExtractionError as exc:
                    logger.warning(
                        "Skipping hand on frame %s (%s).", frame_index, exc
                    )
                    selected_hand = None

                if selected_hand is not None:
                    frames_with_detection += 1
                    if selected_hand.handedness == "Right":
                        right_hand_selected_count += 1
                    elif selected_hand.handedness == "Left":
                        left_hand_selected_count += 1

            detections_frames.append(
                MediaPipeFrameRaw(
                    frame_index=frame_index,
                    timestamp_sec=timestamp_sec,
                    has_detection=selected_hand is not None,
                    num_hands_detected=num_hands_detected,
                    selected_hand=selected_hand,
                )
            )

            frame_index += 1
            total_frames_processed = frame_index
    finally:
        capture.release()
        close = getattr(detector, "close", None)
        if callable(close):
            close()

    # Some containers report an unreliable ``CAP_PROP_FRAME_COUNT``. Prefer
    # the number of frames we actually read so downstream consumers always
    # see a consistent value.
    effective_frame_count = max(total_frames_processed, frame_count)

    detection_rate = (
        frames_with_detection / total_frames_processed
        if total_frames_processed > 0
        else 0.0
    )

    detections_document = MediaPipeDetectionsDocument(
        run_id=effective_run_id,
        source_video_path=str(source_path),
        fps=fps,
        frame_count=effective_frame_count,
        width=width,
        height=height,
        selected_hand_policy=selected_hand_policy,
        frames=detections_frames,
    )

    metadata_document = MediaPipeRunMeta(
        run_id=effective_run_id,
        source_video_path=str(source_path),
        fps=fps,
        frame_count=effective_frame_count,
        width=width,
        height=height,
        created_at=datetime.now(timezone.utc).isoformat(),
        selected_hand_policy=selected_hand_policy,
        total_frames=total_frames_processed,
        frames_with_detection=frames_with_detection,
        detection_rate=detection_rate,
        right_hand_selected_count=right_hand_selected_count,
        left_hand_selected_count=left_hand_selected_count,
    )

    detections_path = run_dir / "detections.json"
    metadata_path = run_dir / "metadata.json"
    _write_json(detections_path, detections_document)
    _write_json(metadata_path, metadata_document)

    logger.info(
        "MediaPipe extraction complete: run_id=%s frames=%s detected=%s (%.2f%%)",
        effective_run_id,
        total_frames_processed,
        frames_with_detection,
        detection_rate * 100.0,
    )

    return MediaPipeExtractionResult(
        run_id=effective_run_id,
        run_dir=run_dir,
        detections_path=detections_path,
        metadata_path=metadata_path,
        detections=detections_document,
        metadata=metadata_document,
    )
