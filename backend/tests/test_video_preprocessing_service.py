from __future__ import annotations

from pathlib import Path

from app.core.config import settings
from app.services.video_preprocessing_service import (
    extract_and_normalize_video,
    prepare_videos_for_analysis,
)


def _pottery_video_path() -> Path:
    return settings.STORAGE_ROOT / "expert" / "pottery.mp4"


def test_extract_and_normalize_video_returns_structured_frames() -> None:
    output = extract_and_normalize_video(_pottery_video_path())

    assert output.video_path.endswith("pottery.mp4")
    assert output.source_fps > 0
    assert output.fps == settings.STANDARDIZATION_TARGET_FPS
    assert output.total_frames == len(output.frames)
    assert output.total_frames > 0
    assert output.total_frames == round(output.duration_seconds * settings.STANDARDIZATION_TARGET_FPS)
    assert output.width == settings.STANDARDIZATION_FRAME_WIDTH
    assert output.height == settings.STANDARDIZATION_FRAME_HEIGHT
    assert output.frames[0].frame_index == 0
    assert output.frames[0].source_frame_index == 0
    assert output.frames[0].timestamp_sec == 0.0
    assert output.frames[0].frame_rgb.shape == (
        settings.STANDARDIZATION_FRAME_HEIGHT,
        settings.STANDARDIZATION_FRAME_WIDTH,
        3,
    )

    timestamps = [frame.timestamp_sec for frame in output.frames]
    assert timestamps == sorted(timestamps)
    assert timestamps[1] == 1 / settings.STANDARDIZATION_TARGET_FPS
    assert timestamps[2] == 2 / settings.STANDARDIZATION_TARGET_FPS

    for frame in output.frames[:10]:
        assert frame.frame_rgb.shape == (
            settings.STANDARDIZATION_FRAME_HEIGHT,
            settings.STANDARDIZATION_FRAME_WIDTH,
            3,
        )


def test_prepare_videos_for_analysis_normalizes_both_inputs() -> None:
    prepared = prepare_videos_for_analysis(
        expert_video_path=_pottery_video_path(),
        learner_video_path=_pottery_video_path(),
    )

    assert prepared.target_fps == settings.STANDARDIZATION_TARGET_FPS
    assert prepared.target_width == settings.STANDARDIZATION_FRAME_WIDTH
    assert prepared.target_height == settings.STANDARDIZATION_FRAME_HEIGHT
    assert prepared.expert_video.total_frames > 0
    assert prepared.learner_video.total_frames > 0
    assert prepared.expert_video.fps == settings.STANDARDIZATION_TARGET_FPS
    assert prepared.learner_video.fps == settings.STANDARDIZATION_TARGET_FPS
    assert prepared.expert_video.width == prepared.learner_video.width
    assert prepared.expert_video.height == prepared.learner_video.height
