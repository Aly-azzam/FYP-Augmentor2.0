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
from typing import Dict, List, Optional

from app.schemas.mediapipe.mediapipe_schema import (
    MediaPipeDetectionsDocument,
    MediaPipeFeaturesDocument,
    MediaPipeFrameFeatures,
    MediaPipeHandFrameFeatures,
    MediaPipeHandBoundingBox,
    MediaPipeJointAngles,
    MediaPipeSelectedHandRaw,
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

def _landmark_vectors_from_hand(hand: MediaPipeSelectedHandRaw) -> List[List[float]]:
    return [[lm.x, lm.y, lm.z] for lm in hand.landmarks]


def _point3(values: List[float]) -> List[float]:
    if len(values) >= 3:
        return [float(values[0]), float(values[1]), float(values[2])]
    if len(values) == 2:
        return [float(values[0]), float(values[1]), 0.0]
    return [0.0, 0.0, 0.0]


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


def _select_best_hand_by_label(
    hands: List[MediaPipeSelectedHandRaw],
) -> Dict[str, MediaPipeSelectedHandRaw]:
    selected: Dict[str, MediaPipeSelectedHandRaw] = {}
    for hand in hands:
        if hand.handedness not in {"Left", "Right"}:
            continue
        current = selected.get(hand.handedness)
        if current is None or hand.detection_confidence > current.detection_confidence:
            selected[hand.handedness] = hand
    return selected


def _build_hand_features(
    *,
    hand: MediaPipeSelectedHandRaw,
    timestamp_sec: float,
    detected: bool,
    fallback_used: bool,
    previous_points: Optional[Dict[str, List[float]]],
    previous_timestamp: Optional[float],
    wrist_trajectory: List[List[float]],
    trajectory_history_size: int,
) -> MediaPipeHandFrameFeatures:
    landmarks = _landmark_vectors_from_hand(hand)
    wrist_xyz = _point3(hand.wrist)
    index_tip = _point3(hand.index_tip)
    thumb_tip = _point3(hand.thumb_tip)
    middle_tip = _point3(hand.middle_tip)
    hand_center = _point3(hand.hand_center)

    bbox_model = MediaPipeHandBoundingBox(**compute_bbox_from_landmarks(landmarks))
    orientation_deg = compute_hand_orientation(landmarks)
    wrist_relative = _compute_wrist_relative_landmarks(landmarks)
    joint_angles = _compute_joint_angles(landmarks)

    if previous_timestamp is not None:
        dt = max(timestamp_sec - previous_timestamp, 0.0)
    else:
        dt = 0.0

    previous_points = previous_points or {}
    wrist_velocity = compute_velocity(wrist_xyz, previous_points.get("wrist"), dt)
    index_tip_velocity = compute_velocity(index_tip, previous_points.get("index_tip"), dt)
    thumb_tip_velocity = compute_velocity(thumb_tip, previous_points.get("thumb_tip"), dt)
    middle_tip_velocity = compute_velocity(middle_tip, previous_points.get("middle_tip"), dt)

    wrist_trajectory.append([float(wrist_xyz[0]), float(wrist_xyz[1])])

    return MediaPipeHandFrameFeatures(
        handedness=hand.handedness,
        detected=detected,
        fallback_used=fallback_used,
        confidence=hand.detection_confidence if detected else max(hand.detection_confidence * 0.5, 0.01),
        wrist=wrist_xyz,
        index_tip=index_tip,
        thumb_tip=thumb_tip,
        middle_tip=middle_tip,
        hand_center=hand_center,
        landmarks=landmarks,
        hand_bbox=bbox_model,
        wrist_angle_deg=orientation_deg,
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

    last_good_hand: Dict[str, MediaPipeSelectedHandRaw] = {}
    previous_points_by_hand: Dict[str, Dict[str, List[float]]] = {}
    previous_timestamp_by_hand: Dict[str, float] = {}
    wrist_trajectory_by_hand: Dict[str, List[List[float]]] = {
        "Left": [],
        "Right": [],
    }
    feature_frames: List[MediaPipeFrameFeatures] = []

    for raw_frame in detections.frames:
        source_hands = raw_frame.hands or (
            [raw_frame.selected_hand] if raw_frame.selected_hand is not None else []
        )
        detected_by_label = _select_best_hand_by_label(source_hands)
        hand_features_by_label: Dict[str, MediaPipeHandFrameFeatures] = {}

        for label in ("Left", "Right"):
            detected_hand = detected_by_label.get(label)
            source_hand = detected_hand or last_good_hand.get(label)
            if source_hand is None:
                continue

            trajectory = wrist_trajectory_by_hand.setdefault(label, [])
            hand_feature = _build_hand_features(
                hand=source_hand,
                timestamp_sec=raw_frame.timestamp_sec,
                detected=detected_hand is not None,
                fallback_used=detected_hand is None,
                previous_points=previous_points_by_hand.get(label),
                previous_timestamp=previous_timestamp_by_hand.get(label),
                wrist_trajectory=trajectory,
                trajectory_history_size=trajectory_history_size,
            )
            hand_features_by_label[label] = hand_feature

            previous_points_by_hand[label] = {
                "wrist": hand_feature.wrist,
                "index_tip": hand_feature.index_tip,
                "thumb_tip": hand_feature.thumb_tip,
                "middle_tip": hand_feature.middle_tip,
            }
            previous_timestamp_by_hand[label] = raw_frame.timestamp_sec
            if detected_hand is not None:
                last_good_hand[label] = detected_hand

        hand_features = [
            hand_features_by_label[label]
            for label in ("Left", "Right")
            if label in hand_features_by_label
        ]
        left_hand_features = hand_features_by_label.get("Left")
        right_hand_features = hand_features_by_label.get("Right")
        selected_feature = right_hand_features or left_hand_features
        has_detection = bool(raw_frame.hands)

        feature_frames.append(
            MediaPipeFrameFeatures(
                frame_index=raw_frame.frame_index,
                timestamp_sec=raw_frame.timestamp_sec,
                has_detection=has_detection,
                handedness=selected_feature.handedness if selected_feature else None,
                hands=hand_features,
                left_hand_features=left_hand_features,
                right_hand_features=right_hand_features,
                wrist=selected_feature.wrist if selected_feature else None,
                hand_center=selected_feature.hand_center if selected_feature else None,
                hand_bbox=selected_feature.hand_bbox if selected_feature else None,
                hand_orientation_deg=(
                    selected_feature.hand_orientation_deg if selected_feature else None
                ),
                wrist_relative_landmarks=(
                    selected_feature.wrist_relative_landmarks if selected_feature else None
                ),
                joint_angles=selected_feature.joint_angles if selected_feature else None,
                wrist_velocity=selected_feature.wrist_velocity if selected_feature else None,
                index_tip_velocity=(
                    selected_feature.index_tip_velocity if selected_feature else None
                ),
                thumb_tip_velocity=(
                    selected_feature.thumb_tip_velocity if selected_feature else None
                ),
                middle_tip_velocity=(
                    selected_feature.middle_tip_velocity if selected_feature else None
                ),
                trajectory_history=(
                    selected_feature.trajectory_history if selected_feature else []
                ),
            )
        )

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
