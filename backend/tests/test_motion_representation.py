import asyncio
import math
from pathlib import Path

from app.core.config import settings
from app.schemas.landmark_schema import FrameLandmarks, HandLandmarks, LandmarkPoint, PerceptionOutput
from app.services.landmark_cleaning_service import clean_video_landmarks
from app.services.motion_representation_service import (
    MotionRepresentationService,
    process_video_to_motion_representation,
    run_motion_representation,
)
from app.services.video_preprocessing_service import extract_and_normalize_video
from app.services.hand_detection_service import extract_hand_landmarks


def _pottery_video_path() -> Path:
    return settings.STORAGE_ROOT / "expert" / "pottery.mp4"


def _make_hand(x: float, y: float, z: float = 0.0) -> HandLandmarks:
    return HandLandmarks(
        landmarks=[
            LandmarkPoint(x=x + landmark_index * 0.001, y=y + landmark_index * 0.001, z=z)
            for landmark_index in range(21)
        ]
    )


def test_process_video_to_motion_representation_returns_consistent_features() -> None:
    output = process_video_to_motion_representation(_pottery_video_path())

    assert output.total_frames > 0
    assert output.sequence_length == output.total_frames
    assert output.frame_feature_vector_dim == 108
    assert output.hand_feature_vector_dim == 54
    assert len(output.frames) == output.total_frames
    assert all(len(frame.flattened_feature_vector) == output.frame_feature_vector_dim for frame in output.frames)

    detected_frame = next(
        frame for frame in output.frames
        if frame.left_hand_features.present or frame.right_hand_features.present
    )
    sample_hand = detected_frame.left_hand_features if detected_frame.left_hand_features.present else detected_frame.right_hand_features

    assert math.isfinite(sample_hand.hand_openness)
    assert math.isfinite(sample_hand.pinch_distance)
    assert math.isfinite(sample_hand.finger_spread)
    assert all(math.isfinite(value) for value in sample_hand.relative_distances.values())
    assert all(math.isfinite(value) for value in sample_hand.joint_angles.values())


def test_motion_representation_preserves_frame_count_and_timestamps() -> None:
    cleaned_landmarks = clean_video_landmarks(
        extract_hand_landmarks(extract_and_normalize_video(_pottery_video_path()))
    )
    service = MotionRepresentationService()
    output = service.build_motion_representation(cleaned_landmarks)

    assert output.total_frames == cleaned_landmarks.total_frames
    assert [frame.timestamp_sec for frame in output.frames[:10]] == [
        frame.timestamp_sec for frame in cleaned_landmarks.frames[:10]
    ]


def test_motion_representation_handles_missing_hands_with_fixed_vector_size() -> None:
    perception_output = PerceptionOutput(
        video_id="demo",
        fps=30.0,
        total_frames=3,
        frames=[
            FrameLandmarks(frame_index=0, timestamp_sec=0.0, left_hand=None, right_hand=None),
            FrameLandmarks(frame_index=1, timestamp_sec=1 / 30.0, left_hand=_make_hand(0.2, 0.3), right_hand=None),
            FrameLandmarks(frame_index=2, timestamp_sec=2 / 30.0, left_hand=None, right_hand=None),
        ],
        video_path="demo.mp4",
        coordinate_system="normalized",
        vjepa_features=None,
    )

    service = MotionRepresentationService()
    output = service.build_motion_representation(perception_output)

    assert len(output.frames[0].flattened_feature_vector) == output.frame_feature_vector_dim
    assert output.frames[0].left_hand_features.present is False
    assert output.frames[0].right_hand_features.present is False
    assert all(value == 0.0 for value in output.frames[0].left_hand_features.flattened_feature_vector)
    assert output.frames[1].left_hand_features.present is True


def test_run_motion_representation_builds_both_outputs() -> None:
    perception_output = PerceptionOutput(
        video_id="demo",
        fps=30.0,
        total_frames=1,
        frames=[FrameLandmarks(frame_index=0, timestamp_sec=0.0, left_hand=_make_hand(0.2, 0.3), right_hand=None)],
        video_path="demo.mp4",
        coordinate_system="normalized",
        vjepa_features=None,
    )

    result = asyncio.run(
        run_motion_representation(
            expert_perception=perception_output,
            learner_perception=perception_output,
        )
    )

    assert set(result.keys()) == {"expert_motion", "learner_motion", "alignment"}
    assert result["alignment"] == {}
    assert result["expert_motion"].frame_feature_vector_dim == 108
    assert result["learner_motion"].sequence_length == 1
