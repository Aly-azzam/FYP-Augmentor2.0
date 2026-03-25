from __future__ import annotations

from pathlib import Path

import numpy as np

from app.core.config import settings
from app.schemas.video_preprocessing_schema import VideoFrame, VideoFramesOutput
from app.services.hand_detection_service import (
    extract_hand_landmarks,
    process_video_to_landmarks,
)
from app.services.video_preprocessing_service import extract_and_normalize_video


def _pottery_video_path() -> Path:
    return settings.STORAGE_ROOT / "expert" / "pottery.mp4"


def test_extract_hand_landmarks_returns_structured_output() -> None:
    video_frames = extract_and_normalize_video(_pottery_video_path())
    output = extract_hand_landmarks(video_frames)

    assert output.video_path is not None
    assert output.coordinate_system == "normalized"
    assert output.fps == video_frames.fps
    assert output.total_frames == video_frames.total_frames
    assert len(output.frames) == video_frames.total_frames
    assert output.frames[0].timestamp_sec == video_frames.frames[0].timestamp_sec

    detected_frames = [
        frame for frame in output.frames if frame.left_hand is not None or frame.right_hand is not None
    ]
    assert detected_frames, "Expected at least one frame with detected hands."

    sample_frame = detected_frames[0]
    sample_hand = sample_frame.left_hand or sample_frame.right_hand
    assert sample_hand is not None
    assert len(sample_hand.landmarks) == 21


def test_extract_hand_landmarks_handles_missing_detections() -> None:
    blank_frame = np.zeros(
        (
            settings.STANDARDIZATION_FRAME_HEIGHT,
            settings.STANDARDIZATION_FRAME_WIDTH,
            3,
        ),
        dtype=np.uint8,
    )

    video_frames = VideoFramesOutput(
        video_path="blank.mp4",
        source_fps=30.0,
        fps=30.0,
        source_width=settings.STANDARDIZATION_FRAME_WIDTH,
        source_height=settings.STANDARDIZATION_FRAME_HEIGHT,
        width=settings.STANDARDIZATION_FRAME_WIDTH,
        height=settings.STANDARDIZATION_FRAME_HEIGHT,
        source_total_frames=3,
        total_frames=3,
        duration_seconds=0.1,
        frames=[
            VideoFrame(frame_index=i, source_frame_index=i, timestamp_sec=i / 30.0, frame_rgb=blank_frame)
            for i in range(3)
        ],
    )

    output = extract_hand_landmarks(video_frames)

    assert len(output.frames) == 3
    assert all(frame.left_hand is None and frame.right_hand is None for frame in output.frames)


def test_process_video_to_landmarks_keeps_preprocessing_timestamps() -> None:
    preprocessed = extract_and_normalize_video(_pottery_video_path())
    output = process_video_to_landmarks(_pottery_video_path())

    assert output.fps == settings.STANDARDIZATION_TARGET_FPS
    assert len(output.frames) == len(preprocessed.frames)
    assert [frame.timestamp_sec for frame in output.frames[:5]] == [
        frame.timestamp_sec for frame in preprocessed.frames[:5]
    ]
