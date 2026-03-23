"""Landmark schema (Perception Engine placeholders).

This file defines lightweight placeholder schema classes that will later
represent per-frame hand landmarks and the overall video-level output.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from typing import List


@dataclass
class LandmarkPoint:
    """A single landmark point (normalized coordinates)."""

    x: float
    y: float
    z: Optional[float] = None
    confidence: Optional[float] = None


@dataclass
class HandLandmarks:
    """A collection of hand landmarks for one hand in one frame."""

    landmarks: List[LandmarkPoint] = field(default_factory=list)


@dataclass
class FrameLandmarks:
    """Hand landmarks for a single frame."""

    frame_index: int
    timestamp_sec: float
    left_hand: Optional[HandLandmarks] = None
    right_hand: Optional[HandLandmarks] = None


@dataclass
class VideoLandmarksOutput:
    """Full output contract of the Perception Engine (placeholder)."""

    video_id: str
    fps: float
    total_frames: int
    frames: List[FrameLandmarks] = field(default_factory=list)
    vjepa_features: dict | None = None


# Backwards compatibility for older module type hints.
PerceptionOutput = VideoLandmarksOutput
