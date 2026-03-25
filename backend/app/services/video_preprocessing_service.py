"""Video standardization service for Phase 2 Step 2.1.

This module is responsible only for deterministic video preparation:
- opening stored expert/learner videos
- extracting frame sequences
- resampling every video to a fixed target FPS
- resizing every kept frame to a fixed resolution
- converting frames from OpenCV BGR to RGB

It does not perform landmark extraction, scoring, or any other CV logic.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2

from app.core.config import settings
from app.schemas.video_preprocessing_schema import (
    PreparedVideosOutput,
    VideoFrame,
    VideoFramesOutput,
)


class VideoPreprocessingError(RuntimeError):
    """Base error for video preprocessing failures."""


class VideoOpenError(VideoPreprocessingError):
    """Raised when a video file cannot be opened."""


class EmptyVideoError(VideoPreprocessingError):
    """Raised when a video contains no readable frames."""


class CorruptedVideoError(VideoPreprocessingError):
    """Raised when a video stops before any useful frames are extracted."""


@dataclass(frozen=True)
class _VideoMetadata:
    video_path: str
    fps: float
    total_frames: int
    width: int
    height: int
    duration_seconds: float


def _resolve_video_path(video_path: str | Path) -> Path:
    resolved = Path(video_path).expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"Video file not found: {resolved}")
    return resolved


def _open_video_capture(video_path: Path) -> cv2.VideoCapture:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        capture.release()
        raise VideoOpenError(f"Could not open video for reading: {video_path}")
    return capture


def _read_metadata(capture: cv2.VideoCapture, video_path: Path) -> _VideoMetadata:
    raw_fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
    source_fps = raw_fps if raw_fps > 0 else settings.STANDARDIZATION_TARGET_FPS

    total_frames = int(float(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0))
    width = int(float(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0.0))
    height = int(float(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0.0))
    duration_seconds = (total_frames / source_fps) if total_frames > 0 and source_fps > 0 else 0.0

    return _VideoMetadata(
        video_path=str(video_path),
        fps=source_fps,
        total_frames=total_frames,
        width=width,
        height=height,
        duration_seconds=duration_seconds,
    )


def _normalize_frame(frame_bgr, target_size: tuple[int, int]):
    width, height = target_size
    resized = cv2.resize(frame_bgr, (width, height), interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)


def _read_source_frames(
    capture: cv2.VideoCapture,
    metadata: _VideoMetadata,
    target_size: tuple[int, int],
) -> list:
    source_frames = []

    while True:
        success, frame_bgr = capture.read()
        if not success:
            break
        source_frames.append(_normalize_frame(frame_bgr, target_size))

    if not source_frames:
        raise EmptyVideoError(f"Video contains no readable frames: {metadata.video_path}")

    return source_frames


def _compute_target_total_frames(metadata: _VideoMetadata, target_fps: float) -> int:
    if metadata.duration_seconds > 0:
        return max(1, int(round(metadata.duration_seconds * target_fps)))

    # Fallback when metadata duration is unavailable but readable frames exist.
    return max(1, int(round(metadata.total_frames * (target_fps / metadata.fps))))


def _resample_frames_to_target_fps(
    source_frames: list,
    metadata: _VideoMetadata,
    target_fps: float,
) -> list[VideoFrame]:
    target_total_frames = _compute_target_total_frames(metadata, target_fps)
    last_source_index = len(source_frames) - 1
    normalized_frames: list[VideoFrame] = []

    for target_frame_index in range(target_total_frames):
        target_timestamp_sec = target_frame_index / target_fps
        mapped_source_index = round(target_timestamp_sec * metadata.fps)
        clamped_source_index = max(0, min(mapped_source_index, last_source_index))

        normalized_frames.append(
            VideoFrame(
                frame_index=target_frame_index,
                source_frame_index=clamped_source_index,
                timestamp_sec=target_timestamp_sec,
                frame_rgb=source_frames[clamped_source_index],
            )
        )

    return normalized_frames


def _build_frames_output(
    metadata: _VideoMetadata,
    frames: list[VideoFrame],
    target_fps: float,
    target_size: tuple[int, int],
) -> VideoFramesOutput:
    width, height = target_size

    return VideoFramesOutput(
        video_path=metadata.video_path,
        source_fps=metadata.fps,
        fps=target_fps,
        source_width=metadata.width,
        source_height=metadata.height,
        width=width,
        height=height,
        source_total_frames=metadata.total_frames,
        total_frames=len(frames),
        duration_seconds=metadata.duration_seconds,
        frames=frames,
    )


def extract_and_normalize_video(
    video_path: str | Path,
    *,
    target_fps: float = settings.STANDARDIZATION_TARGET_FPS,
    target_size: tuple[int, int] = (
        settings.STANDARDIZATION_FRAME_WIDTH,
        settings.STANDARDIZATION_FRAME_HEIGHT,
    ),
) -> VideoFramesOutput:
    """Extract a deterministic normalized RGB frame sequence for one video."""
    if target_fps <= 0:
        raise ValueError("target_fps must be greater than zero.")

    target_width, target_height = target_size
    if target_width <= 0 or target_height <= 0:
        raise ValueError("target_size values must be greater than zero.")

    resolved_path = _resolve_video_path(video_path)
    capture = _open_video_capture(resolved_path)

    try:
        metadata = _read_metadata(capture, resolved_path)
        source_frames = _read_source_frames(capture, metadata, target_size)
        normalized_frames = _resample_frames_to_target_fps(source_frames, metadata, target_fps)

        if not normalized_frames:
            raise CorruptedVideoError(
                f"Video frames could not be extracted after normalization: {resolved_path}"
            )

        return _build_frames_output(metadata, normalized_frames, target_fps, target_size)
    finally:
        capture.release()


def prepare_videos_for_analysis(
    expert_video_path: str | Path,
    learner_video_path: str | Path,
    *,
    target_fps: float = settings.STANDARDIZATION_TARGET_FPS,
    target_size: tuple[int, int] = (
        settings.STANDARDIZATION_FRAME_WIDTH,
        settings.STANDARDIZATION_FRAME_HEIGHT,
    ),
) -> PreparedVideosOutput:
    """Normalize expert and learner videos to the same target settings."""
    expert_video = extract_and_normalize_video(
        expert_video_path,
        target_fps=target_fps,
        target_size=target_size,
    )
    learner_video = extract_and_normalize_video(
        learner_video_path,
        target_fps=target_fps,
        target_size=target_size,
    )

    return PreparedVideosOutput(
        target_fps=target_fps,
        target_width=target_size[0],
        target_height=target_size[1],
        expert_video=expert_video,
        learner_video=learner_video,
    )
