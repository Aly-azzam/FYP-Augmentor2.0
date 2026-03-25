"""Motion Representation Service — build comparison-ready hand-motion features."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from app.schemas.landmark_schema import HandLandmarks, VideoLandmarksOutput
from app.schemas.motion_schema import (
    HandMotionFeatures,
    MotionFrameFeatures,
    MotionOutput,
    MotionRepresentationOutput,
)
from app.services.landmark_cleaning_service import process_video_to_clean_landmarks
from app.utils.math_utils import angle_between_vectors, euclidean_distance, zero_vector

FINGERTIP_INDICES = {
    "thumb_tip": 4,
    "index_tip": 8,
    "middle_tip": 12,
    "ring_tip": 16,
    "pinky_tip": 20,
}
PALM_CENTER_INDICES = [0, 5, 9, 13, 17]
WRIST_INDEX = 0
INDEX_MCP_INDEX = 5
MIDDLE_MCP_INDEX = 9
RING_MCP_INDEX = 13
PINKY_MCP_INDEX = 17
HAND_SCALE_EPSILON = 1e-6


def _point_to_vector(point) -> list[float]:
    return [float(point.x), float(point.y), float(point.z if point.z is not None else 0.0)]


def _vector_subtract(a: list[float], b: list[float]) -> list[float]:
    return [a[index] - b[index] for index in range(len(a))]


def _vector_divide(vector: list[float], scalar: float) -> list[float]:
    if abs(scalar) < HAND_SCALE_EPSILON:
        return zero_vector(len(vector))
    return [value / scalar for value in vector]


def _compute_palm_center(hand: HandLandmarks) -> list[float]:
    vectors = [_point_to_vector(hand.landmarks[index]) for index in PALM_CENTER_INDICES]
    return [sum(values) / len(values) for values in zip(*vectors)]


def _compute_hand_scale(hand: HandLandmarks) -> float:
    wrist = _point_to_vector(hand.landmarks[WRIST_INDEX])
    middle_mcp = _point_to_vector(hand.landmarks[MIDDLE_MCP_INDEX])
    return max(euclidean_distance(wrist, middle_mcp), HAND_SCALE_EPSILON)


def _compute_angle(hand: HandLandmarks, index_a: int, index_b: int, index_c: int) -> float:
    point_a = _point_to_vector(hand.landmarks[index_a])
    point_b = _point_to_vector(hand.landmarks[index_b])
    point_c = _point_to_vector(hand.landmarks[index_c])
    vector_ab = _vector_subtract(point_a, point_b)
    vector_cb = _vector_subtract(point_c, point_b)
    return float(angle_between_vectors(vector_ab, vector_cb))


def _compute_temporal_vector(
    current: list[float],
    previous: Optional[list[float]],
    dt: float,
) -> list[float]:
    if previous is None or dt <= 0:
        return zero_vector(len(current))
    return [(current[index] - previous[index]) / dt for index in range(len(current))]


class MotionRepresentationService:
    """Build interpretable, deterministic motion features from cleaned landmarks."""

    def __init__(self) -> None:
        self.hand_feature_vector_dim = 54
        self.frame_feature_vector_dim = self.hand_feature_vector_dim * 2

    def build_motion_representation(
        self,
        video_landmarks: VideoLandmarksOutput,
    ) -> MotionRepresentationOutput:
        left_prev_wrist: Optional[list[float]] = None
        left_prev_palm: Optional[list[float]] = None
        left_prev_wrist_velocity: Optional[list[float]] = None
        left_prev_palm_velocity: Optional[list[float]] = None

        right_prev_wrist: Optional[list[float]] = None
        right_prev_palm: Optional[list[float]] = None
        right_prev_wrist_velocity: Optional[list[float]] = None
        right_prev_palm_velocity: Optional[list[float]] = None

        previous_timestamp: Optional[float] = None
        output_frames: list[MotionFrameFeatures] = []

        for frame in video_landmarks.frames:
            dt = 0.0 if previous_timestamp is None else frame.timestamp_sec - previous_timestamp

            left_features, left_prev_wrist, left_prev_palm, left_prev_wrist_velocity, left_prev_palm_velocity = (
                self._build_hand_features(
                    hand=frame.left_hand,
                    dt=dt,
                    previous_wrist=left_prev_wrist,
                    previous_palm=left_prev_palm,
                    previous_wrist_velocity=left_prev_wrist_velocity,
                    previous_palm_velocity=left_prev_palm_velocity,
                )
            )
            right_features, right_prev_wrist, right_prev_palm, right_prev_wrist_velocity, right_prev_palm_velocity = (
                self._build_hand_features(
                    hand=frame.right_hand,
                    dt=dt,
                    previous_wrist=right_prev_wrist,
                    previous_palm=right_prev_palm,
                    previous_wrist_velocity=right_prev_wrist_velocity,
                    previous_palm_velocity=right_prev_palm_velocity,
                )
            )

            output_frames.append(
                MotionFrameFeatures(
                    frame_index=frame.frame_index,
                    timestamp_sec=frame.timestamp_sec,
                    left_hand_features=left_features,
                    right_hand_features=right_features,
                    flattened_feature_vector=(
                        left_features.flattened_feature_vector + right_features.flattened_feature_vector
                    ),
                )
            )
            previous_timestamp = frame.timestamp_sec

        return MotionRepresentationOutput(
            video_id=video_landmarks.video_id,
            video_path=video_landmarks.video_path,
            coordinate_system=video_landmarks.coordinate_system,
            fps=video_landmarks.fps,
            total_frames=video_landmarks.total_frames,
            sequence_length=video_landmarks.total_frames,
            hand_feature_vector_dim=self.hand_feature_vector_dim,
            frame_feature_vector_dim=self.frame_feature_vector_dim,
            frames=output_frames,
        )

    def _build_hand_features(
        self,
        *,
        hand: Optional[HandLandmarks],
        dt: float,
        previous_wrist: Optional[list[float]],
        previous_palm: Optional[list[float]],
        previous_wrist_velocity: Optional[list[float]],
        previous_palm_velocity: Optional[list[float]],
    ) -> tuple[HandMotionFeatures, Optional[list[float]], Optional[list[float]], Optional[list[float]], Optional[list[float]]]:
        zero3 = zero_vector(3)
        zero_tip_positions = {name: zero3[:] for name in FINGERTIP_INDICES}
        zero_distances = {
            "thumb_tip_to_index_tip": 0.0,
            "thumb_tip_to_middle_tip": 0.0,
            "index_tip_to_wrist": 0.0,
            "middle_tip_to_wrist": 0.0,
            "thumb_tip_to_palm_center": 0.0,
            "index_tip_to_palm_center": 0.0,
            "middle_tip_to_palm_center": 0.0,
            "ring_tip_to_palm_center": 0.0,
            "pinky_tip_to_palm_center": 0.0,
            "index_tip_to_pinky_tip": 0.0,
        }
        zero_angles = {
            "wrist_index_mcp_index_tip": 0.0,
            "wrist_middle_mcp_middle_tip": 0.0,
            "wrist_ring_mcp_ring_tip": 0.0,
            "wrist_index_mcp_pinky_mcp": 0.0,
        }

        if hand is None:
            features = HandMotionFeatures(
                present=False,
                wrist_position=zero3[:],
                palm_center=zero3[:],
                hand_scale=0.0,
                wrist_relative_to_palm=zero3[:],
                fingertip_positions={name: value[:] for name, value in zero_tip_positions.items()},
                fingertip_relative_positions={name: value[:] for name, value in zero_tip_positions.items()},
                relative_distances=zero_distances.copy(),
                joint_angles=zero_angles.copy(),
                hand_openness=0.0,
                pinch_distance=0.0,
                finger_spread=0.0,
                wrist_velocity=zero3[:],
                palm_velocity=zero3[:],
                wrist_acceleration=zero3[:],
                palm_acceleration=zero3[:],
                flattened_feature_vector=self._flatten_hand_features(
                    present=False,
                    wrist_position=zero3[:],
                    palm_center=zero3[:],
                    wrist_relative_to_palm=zero3[:],
                    fingertip_relative_positions={name: value[:] for name, value in zero_tip_positions.items()},
                    relative_distances=zero_distances,
                    joint_angles=zero_angles,
                    hand_openness=0.0,
                    pinch_distance=0.0,
                    finger_spread=0.0,
                    wrist_velocity=zero3[:],
                    palm_velocity=zero3[:],
                    wrist_acceleration=zero3[:],
                    palm_acceleration=zero3[:],
                ),
            )
            return features, None, None, None, None

        wrist_position = _point_to_vector(hand.landmarks[WRIST_INDEX])
        palm_center = _compute_palm_center(hand)
        hand_scale = _compute_hand_scale(hand)
        wrist_relative_to_palm = _vector_divide(_vector_subtract(wrist_position, palm_center), hand_scale)

        fingertip_positions = {
            name: _point_to_vector(hand.landmarks[index])
            for name, index in FINGERTIP_INDICES.items()
        }
        fingertip_relative_positions = {
            name: _vector_divide(_vector_subtract(position, palm_center), hand_scale)
            for name, position in fingertip_positions.items()
        }

        relative_distances = {
            "thumb_tip_to_index_tip": euclidean_distance(
                fingertip_positions["thumb_tip"], fingertip_positions["index_tip"]
            ) / hand_scale,
            "thumb_tip_to_middle_tip": euclidean_distance(
                fingertip_positions["thumb_tip"], fingertip_positions["middle_tip"]
            ) / hand_scale,
            "index_tip_to_wrist": euclidean_distance(
                fingertip_positions["index_tip"], wrist_position
            ) / hand_scale,
            "middle_tip_to_wrist": euclidean_distance(
                fingertip_positions["middle_tip"], wrist_position
            ) / hand_scale,
            "thumb_tip_to_palm_center": euclidean_distance(
                fingertip_positions["thumb_tip"], palm_center
            ) / hand_scale,
            "index_tip_to_palm_center": euclidean_distance(
                fingertip_positions["index_tip"], palm_center
            ) / hand_scale,
            "middle_tip_to_palm_center": euclidean_distance(
                fingertip_positions["middle_tip"], palm_center
            ) / hand_scale,
            "ring_tip_to_palm_center": euclidean_distance(
                fingertip_positions["ring_tip"], palm_center
            ) / hand_scale,
            "pinky_tip_to_palm_center": euclidean_distance(
                fingertip_positions["pinky_tip"], palm_center
            ) / hand_scale,
            "index_tip_to_pinky_tip": euclidean_distance(
                fingertip_positions["index_tip"], fingertip_positions["pinky_tip"]
            ) / hand_scale,
        }

        joint_angles = {
            "wrist_index_mcp_index_tip": _compute_angle(hand, WRIST_INDEX, INDEX_MCP_INDEX, FINGERTIP_INDICES["index_tip"]),
            "wrist_middle_mcp_middle_tip": _compute_angle(hand, WRIST_INDEX, MIDDLE_MCP_INDEX, FINGERTIP_INDICES["middle_tip"]),
            "wrist_ring_mcp_ring_tip": _compute_angle(hand, WRIST_INDEX, RING_MCP_INDEX, FINGERTIP_INDICES["ring_tip"]),
            "wrist_index_mcp_pinky_mcp": _compute_angle(hand, WRIST_INDEX, INDEX_MCP_INDEX, PINKY_MCP_INDEX),
        }

        hand_openness = sum(
            relative_distances[key]
            for key in (
                "thumb_tip_to_palm_center",
                "index_tip_to_palm_center",
                "middle_tip_to_palm_center",
                "ring_tip_to_palm_center",
                "pinky_tip_to_palm_center",
            )
        ) / 5.0
        pinch_distance = relative_distances["thumb_tip_to_index_tip"]
        finger_spread = relative_distances["index_tip_to_pinky_tip"]

        wrist_velocity = _compute_temporal_vector(wrist_position, previous_wrist, dt)
        palm_velocity = _compute_temporal_vector(palm_center, previous_palm, dt)
        wrist_acceleration = _compute_temporal_vector(wrist_velocity, previous_wrist_velocity, dt)
        palm_acceleration = _compute_temporal_vector(palm_velocity, previous_palm_velocity, dt)

        flattened = self._flatten_hand_features(
            present=True,
            wrist_position=wrist_position,
            palm_center=palm_center,
            wrist_relative_to_palm=wrist_relative_to_palm,
            fingertip_relative_positions=fingertip_relative_positions,
            relative_distances=relative_distances,
            joint_angles=joint_angles,
            hand_openness=hand_openness,
            pinch_distance=pinch_distance,
            finger_spread=finger_spread,
            wrist_velocity=wrist_velocity,
            palm_velocity=palm_velocity,
            wrist_acceleration=wrist_acceleration,
            palm_acceleration=palm_acceleration,
        )

        features = HandMotionFeatures(
            present=True,
            wrist_position=wrist_position,
            palm_center=palm_center,
            hand_scale=hand_scale,
            wrist_relative_to_palm=wrist_relative_to_palm,
            fingertip_positions=fingertip_positions,
            fingertip_relative_positions=fingertip_relative_positions,
            relative_distances=relative_distances,
            joint_angles=joint_angles,
            hand_openness=hand_openness,
            pinch_distance=pinch_distance,
            finger_spread=finger_spread,
            wrist_velocity=wrist_velocity,
            palm_velocity=palm_velocity,
            wrist_acceleration=wrist_acceleration,
            palm_acceleration=palm_acceleration,
            flattened_feature_vector=flattened,
        )
        return features, wrist_position, palm_center, wrist_velocity, palm_velocity

    def _flatten_hand_features(
        self,
        *,
        present: bool,
        wrist_position: list[float],
        palm_center: list[float],
        wrist_relative_to_palm: list[float],
        fingertip_relative_positions: dict[str, list[float]],
        relative_distances: dict[str, float],
        joint_angles: dict[str, float],
        hand_openness: float,
        pinch_distance: float,
        finger_spread: float,
        wrist_velocity: list[float],
        palm_velocity: list[float],
        wrist_acceleration: list[float],
        palm_acceleration: list[float],
    ) -> list[float]:
        vector = [1.0 if present else 0.0]
        vector.extend(wrist_position)
        vector.extend(palm_center)
        vector.extend(wrist_relative_to_palm)
        for name in ("thumb_tip", "index_tip", "middle_tip", "ring_tip", "pinky_tip"):
            vector.extend(fingertip_relative_positions[name])
        for name in (
            "thumb_tip_to_index_tip",
            "thumb_tip_to_middle_tip",
            "index_tip_to_wrist",
            "middle_tip_to_wrist",
            "thumb_tip_to_palm_center",
            "index_tip_to_palm_center",
            "middle_tip_to_palm_center",
            "ring_tip_to_palm_center",
            "pinky_tip_to_palm_center",
            "index_tip_to_pinky_tip",
        ):
            vector.append(relative_distances[name])
        for name in (
            "wrist_index_mcp_index_tip",
            "wrist_middle_mcp_middle_tip",
            "wrist_ring_mcp_ring_tip",
            "wrist_index_mcp_pinky_mcp",
        ):
            vector.append(joint_angles[name])
        vector.extend([hand_openness, pinch_distance, finger_spread])
        vector.extend(wrist_velocity)
        vector.extend(palm_velocity)
        vector.extend(wrist_acceleration)
        vector.extend(palm_acceleration)
        return [float(value) for value in vector]


def process_video_to_motion_representation(video_path: str | Path) -> MotionRepresentationOutput:
    """Run the full pipeline up to comparison-ready motion features."""
    cleaned_landmarks = process_video_to_clean_landmarks(video_path)
    service = MotionRepresentationService()
    return service.build_motion_representation(cleaned_landmarks)


async def run_motion_representation(
    expert_perception: VideoLandmarksOutput,
    learner_perception: VideoLandmarksOutput,
) -> dict:
    """Compatibility wrapper that builds motion representation for two videos."""
    service = MotionRepresentationService()
    return {
        "expert_motion": service.build_motion_representation(expert_perception),
        "learner_motion": service.build_motion_representation(learner_perception),
        "alignment": {},
    }
