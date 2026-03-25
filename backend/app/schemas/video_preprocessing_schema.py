"""Structured output contracts for video standardization.

These contracts intentionally stay independent from landmark extraction and
later CV steps. They represent only normalized video frame sequences.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

import numpy as np


@dataclass
class VideoFrame:
    """A single normalized RGB frame from a video."""

    frame_index: int
    source_frame_index: int
    timestamp_sec: float
    frame_rgb: np.ndarray


@dataclass
class VideoFramesOutput:
    """Normalized frame sequence for one video."""

    video_path: str
    source_fps: float
    fps: float
    source_width: int
    source_height: int
    width: int
    height: int
    source_total_frames: int
    total_frames: int
    duration_seconds: float
    frames: List[VideoFrame] = field(default_factory=list)


@dataclass
class PreparedVideosOutput:
    """Normalized expert/learner pair ready for later analysis."""

    target_fps: float
    target_width: int
    target_height: int
    expert_video: VideoFramesOutput
    learner_video: VideoFramesOutput
