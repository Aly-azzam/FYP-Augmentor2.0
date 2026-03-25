"""MediaPipe hand detection built on top of standardized video frames.

This service is the Phase 2 Step 2.2 hand-detection layer. It consumes the
normalized frame sequences produced by Step 2.1 and outputs structured
landmarks for left/right hands per frame.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import mediapipe as mp

from app.core.config import settings
from app.schemas.landmark_schema import (
    FrameLandmarks,
    HandLandmarks,
    LandmarkPoint,
    VideoLandmarksOutput,
)
from app.schemas.video_preprocessing_schema import VideoFramesOutput
from app.services.video_preprocessing_service import extract_and_normalize_video


class HandDetectionError(RuntimeError):
    """Base error for MediaPipe hand detection failures."""


def create_hands_detector() -> Any:
    """Create a single reusable MediaPipe HandLandmarker instance."""
    model_path = settings.HAND_LANDMARKER_MODEL_PATH
    if not model_path.is_file():
        raise FileNotFoundError(
            f"Hand landmarker model file not found: {model_path}"
        )

    base_options = mp.tasks.BaseOptions(model_asset_path=str(model_path))
    options = mp.tasks.vision.HandLandmarkerOptions(
        base_options=base_options,
        running_mode=mp.tasks.vision.RunningMode.IMAGE,
        num_hands=settings.HAND_DETECTION_MAX_NUM_HANDS,
        min_hand_detection_confidence=settings.HAND_DETECTION_MIN_DETECTION_CONFIDENCE,
        min_hand_presence_confidence=settings.HAND_DETECTION_MIN_PRESENCE_CONFIDENCE,
        min_tracking_confidence=settings.HAND_DETECTION_MIN_TRACKING_CONFIDENCE,
    )
    return mp.tasks.vision.HandLandmarker.create_from_options(options)


def _to_hand_landmarks(hand_landmarks_obj: Any) -> HandLandmarks:
    landmark_list = getattr(hand_landmarks_obj, "landmark", None)
    if landmark_list is None and isinstance(hand_landmarks_obj, list):
        landmark_list = hand_landmarks_obj

    if not landmark_list or len(landmark_list) != 21:
        raise HandDetectionError(
            "Each detected hand must contain exactly 21 landmarks."
        )

    return HandLandmarks(
        landmarks=[
            LandmarkPoint(
                x=float(landmark.x),
                y=float(landmark.y),
                z=float(landmark.z),
            )
            for landmark in landmark_list
        ]
    )


def _resolve_handedness_label(
    results: Any,
    hand_index: int,
    left_hand: Optional[HandLandmarks],
    right_hand: Optional[HandLandmarks],
) -> str:
    handedness_list = getattr(results, "handedness", None) or []
    if hand_index < len(handedness_list):
        classifications = handedness_list[hand_index] or []
        if classifications:
            label = getattr(classifications[0], "category_name", None)
            if label in ("Left", "Right"):
                return label

    # Deterministic fallback if MediaPipe handedness is missing.
    if left_hand is None:
        return "Left"
    if right_hand is None:
        return "Right"
    return "Right"


def _extract_frame_landmarks(frame_index: int, timestamp_sec: float, results: Any) -> FrameLandmarks:
    left_hand: Optional[HandLandmarks] = None
    right_hand: Optional[HandLandmarks] = None

    hand_landmarks_list = getattr(results, "hand_landmarks", None)
    if hand_landmarks_list:
        for hand_index, hand_landmarks_obj in enumerate(hand_landmarks_list):
            structured_hand = _to_hand_landmarks(hand_landmarks_obj)
            label = _resolve_handedness_label(results, hand_index, left_hand, right_hand)

            if label == "Left":
                left_hand = structured_hand
            else:
                right_hand = structured_hand

    return FrameLandmarks(
        frame_index=frame_index,
        timestamp_sec=timestamp_sec,
        left_hand=left_hand,
        right_hand=right_hand,
    )


def extract_hand_landmarks(video_frames: VideoFramesOutput) -> VideoLandmarksOutput:
    """Run MediaPipe Hands on a normalized video frame sequence."""
    detector = create_hands_detector()
    processed_frames: list[FrameLandmarks] = []

    try:
        for frame in video_frames.frames:
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame.frame_rgb)
            results = detector.detect(mp_image)
            processed_frames.append(
                _extract_frame_landmarks(
                    frame_index=frame.frame_index,
                    timestamp_sec=frame.timestamp_sec,
                    results=results,
                )
            )
    finally:
        close = getattr(detector, "close", None)
        if callable(close):
            close()

    return VideoLandmarksOutput(
        video_id=Path(video_frames.video_path).stem,
        video_path=video_frames.video_path,
        coordinate_system="normalized",
        fps=video_frames.fps,
        total_frames=video_frames.total_frames,
        frames=processed_frames,
        vjepa_features=None,
    )


def process_video_to_landmarks(video_path: str | Path) -> VideoLandmarksOutput:
    """Preprocess a video and then extract hand landmarks."""
    video_frames = extract_and_normalize_video(video_path)
    return extract_hand_landmarks(video_frames)
