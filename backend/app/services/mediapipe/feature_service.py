"""MediaPipe feature service.

Consumes a ``MediaPipeDetectionsDocument`` (produced by
``extraction_service.run_extraction``) and builds a derived-feature
document stored as ``features.json`` in the same run folder:

    backend/storage/mediapipe/runs/<run_id>/features.json

Derived features include:
    * wrist-relative coordinates for all 21 landmarks
    * hand bounding box
    * hand orientation (degrees, image space)
    * selected joint angles (degrees)
    * wrist velocity (support only)
    * fingertip velocities (support only)
    * rolling wrist trajectory history

Velocities are intentionally classified as "support" signals, not the
primary metric; the main geometric signals come from angles and
wrist-relative coordinates.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import List, Optional

from app.schemas.mediapipe.mediapipe_schema import (
    MediaPipeDetectionsDocument,
    MediaPipeFeaturesDocument,
    MediaPipeFrameFeatures,
    MediaPipeFrameRaw,
    MediaPipeHandBoundingBox,
    MediaPipeJointAngles,
)
from app.utils.mediapipe.mediapipe_utils import (
    HAND_LANDMARK_NAMES,
    compute_bbox_from_landmarks,
    compute_hand_orientation,
    compute_joint_angle,
    compute_velocity,
)


logger = logging.getLogger(__name__)

DEFAULT_TRAJECTORY_HISTORY_SIZE = 30

_WRIST_INDEX = 0
_THUMB_MCP_INDEX = 2
_THUMB_IP_INDEX = 3
_THUMB_TIP_INDEX = 4
_INDEX_MCP_INDEX = 5
_INDEX_TIP_INDEX = 8
_MIDDLE_MCP_INDEX = 9
_MIDDLE_TIP_INDEX = 12
_RING_MCP_INDEX = 13
_RING_TIP_INDEX = 16
_PINKY_MCP_INDEX = 17


class MediaPipeFeatureError(RuntimeError):
    """Raised when a features document cannot be built."""


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _landmark_vectors(raw_frame: MediaPipeFrameRaw) -> Optional[List[List[float]]]:
    if not raw_frame.has_detection or raw_frame.selected_hand is None:
        return None
    hand = raw_frame.selected_hand
    return [[lm.x, lm.y, lm.z] for lm in hand.landmarks]


def _compute_wrist_relative_landmarks(
    landmarks: List[List[float]],
) -> dict:
    wrist = landmarks[_WRIST_INDEX]
    relative: dict = {}
    for index, point in enumerate(landmarks):
        name = HAND_LANDMARK_NAMES[index]
        relative[name] = [
            point[0] - wrist[0],
            point[1] - wrist[1],
            point[2] - wrist[2],
        ]
    return relative


def _compute_joint_angles(landmarks: List[List[float]]) -> MediaPipeJointAngles:
    return MediaPipeJointAngles(
        wrist_index_mcp_index_tip=compute_joint_angle(
            landmarks[_WRIST_INDEX],
            landmarks[_INDEX_MCP_INDEX],
            landmarks[_INDEX_TIP_INDEX],
        ),
        wrist_middle_mcp_middle_tip=compute_joint_angle(
            landmarks[_WRIST_INDEX],
            landmarks[_MIDDLE_MCP_INDEX],
            landmarks[_MIDDLE_TIP_INDEX],
        ),
        wrist_ring_mcp_ring_tip=compute_joint_angle(
            landmarks[_WRIST_INDEX],
            landmarks[_RING_MCP_INDEX],
            landmarks[_RING_TIP_INDEX],
        ),
        index_mcp_middle_mcp_pinky_mcp=compute_joint_angle(
            landmarks[_INDEX_MCP_INDEX],
            landmarks[_MIDDLE_MCP_INDEX],
            landmarks[_PINKY_MCP_INDEX],
        ),
        thumb_mcp_thumb_ip_thumb_tip=compute_joint_angle(
            landmarks[_THUMB_MCP_INDEX],
            landmarks[_THUMB_IP_INDEX],
            landmarks[_THUMB_TIP_INDEX],
        ),
    )


def _clone_trajectory_tail(
    history: List[List[float]], max_size: int
) -> List[List[float]]:
    if max_size <= 0:
        return []
    trimmed = history[-max_size:]
    return [[float(x), float(y)] for x, y in trimmed]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_features_document(
    detections: MediaPipeDetectionsDocument,
    *,
    trajectory_history_size: int = DEFAULT_TRAJECTORY_HISTORY_SIZE,
) -> MediaPipeFeaturesDocument:
    """Build a ``MediaPipeFeaturesDocument`` from raw detections."""
    if trajectory_history_size < 1:
        raise ValueError("trajectory_history_size must be >= 1.")

    fps = detections.fps
    dt_seconds = 1.0 / fps if fps and fps > 0 else 0.0

    previous_wrist: Optional[List[float]] = None
    previous_index_tip: Optional[List[float]] = None
    previous_thumb_tip: Optional[List[float]] = None
    previous_middle_tip: Optional[List[float]] = None
    previous_timestamp: Optional[float] = None

    wrist_trajectory: List[List[float]] = []
    feature_frames: List[MediaPipeFrameFeatures] = []

    for raw_frame in detections.frames:
        landmarks = _landmark_vectors(raw_frame)

        if landmarks is None:
            feature_frames.append(
                MediaPipeFrameFeatures(
                    frame_index=raw_frame.frame_index,
                    timestamp_sec=raw_frame.timestamp_sec,
                    has_detection=False,
                    handedness=None,
                    wrist=None,
                    hand_center=None,
                    hand_bbox=None,
                    hand_orientation_deg=None,
                    wrist_relative_landmarks=None,
                    joint_angles=None,
                    wrist_velocity=None,
                    index_tip_velocity=None,
                    thumb_tip_velocity=None,
                    middle_tip_velocity=None,
                    trajectory_history=_clone_trajectory_tail(
                        wrist_trajectory, trajectory_history_size
                    ),
                )
            )
            previous_wrist = None
            previous_index_tip = None
            previous_thumb_tip = None
            previous_middle_tip = None
            previous_timestamp = raw_frame.timestamp_sec
            continue

        selected_hand = raw_frame.selected_hand
        assert selected_hand is not None  # for type-checker; has_detection==True

        wrist_xyz = landmarks[_WRIST_INDEX]
        thumb_tip = landmarks[_THUMB_TIP_INDEX]
        index_tip = landmarks[_INDEX_TIP_INDEX]
        middle_tip = landmarks[_MIDDLE_TIP_INDEX]

        bbox_dict = compute_bbox_from_landmarks(landmarks)
        bbox_model = MediaPipeHandBoundingBox(**bbox_dict)

        orientation_deg = compute_hand_orientation(landmarks)
        wrist_relative = _compute_wrist_relative_landmarks(landmarks)
        joint_angles = _compute_joint_angles(landmarks)

        # dt uses actual timestamp delta so velocity stays honest even if the
        # video contains duplicated / dropped frames.
        if previous_timestamp is not None:
            dt = max(raw_frame.timestamp_sec - previous_timestamp, 0.0)
        else:
            dt = 0.0
        if dt <= 0.0 and dt_seconds > 0.0:
            dt = dt_seconds

        wrist_velocity = compute_velocity(wrist_xyz, previous_wrist, dt)
        index_tip_velocity = compute_velocity(index_tip, previous_index_tip, dt)
        thumb_tip_velocity = compute_velocity(thumb_tip, previous_thumb_tip, dt)
        middle_tip_velocity = compute_velocity(middle_tip, previous_middle_tip, dt)

        wrist_trajectory.append([float(wrist_xyz[0]), float(wrist_xyz[1])])

        feature_frames.append(
            MediaPipeFrameFeatures(
                frame_index=raw_frame.frame_index,
                timestamp_sec=raw_frame.timestamp_sec,
                has_detection=True,
                handedness=selected_hand.handedness,
                wrist=list(wrist_xyz),
                hand_center=list(selected_hand.hand_center),
                hand_bbox=bbox_model,
                hand_orientation_deg=orientation_deg,
                wrist_relative_landmarks=wrist_relative,
                joint_angles=joint_angles,
                wrist_velocity=wrist_velocity,
                index_tip_velocity=index_tip_velocity,
                thumb_tip_velocity=thumb_tip_velocity,
                middle_tip_velocity=middle_tip_velocity,
                trajectory_history=_clone_trajectory_tail(
                    wrist_trajectory, trajectory_history_size
                ),
            )
        )

        previous_wrist = wrist_xyz
        previous_index_tip = index_tip
        previous_thumb_tip = thumb_tip
        previous_middle_tip = middle_tip
        previous_timestamp = raw_frame.timestamp_sec

    return MediaPipeFeaturesDocument(
        run_id=detections.run_id,
        source_video_path=detections.source_video_path,
        fps=detections.fps,
        frame_count=detections.frame_count,
        trajectory_history_size=trajectory_history_size,
        frames=feature_frames,
    )


def run_feature_extraction(
    run_dir: str | Path,
    *,
    trajectory_history_size: int = DEFAULT_TRAJECTORY_HISTORY_SIZE,
) -> Path:
    """Read ``detections.json`` from ``run_dir`` and write ``features.json``."""
    run_path = Path(run_dir).resolve()
    detections_path = run_path / "detections.json"
    features_path = run_path / "features.json"

    if not detections_path.is_file():
        raise MediaPipeFeatureError(
            f"detections.json not found in run folder: {detections_path}"
        )

    with detections_path.open("r", encoding="utf-8") as handle:
        detections_payload = json.load(handle)

    try:
        detections_document = MediaPipeDetectionsDocument.model_validate(detections_payload)
    except Exception as exc:  # noqa: BLE001 - wrap for a clearer error surface
        raise MediaPipeFeatureError(
            f"Invalid detections.json at {detections_path}: {exc}"
        ) from exc

    features_document = build_features_document(
        detections_document,
        trajectory_history_size=trajectory_history_size,
    )

    features_path.parent.mkdir(parents=True, exist_ok=True)
    with features_path.open("w", encoding="utf-8") as handle:
        json.dump(features_document.model_dump(mode="json"), handle, ensure_ascii=False, indent=2)

    logger.info(
        "MediaPipe features written: run_id=%s frames=%s",
        features_document.run_id,
        len(features_document.frames),
    )
    return features_path
