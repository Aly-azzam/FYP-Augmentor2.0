"""Landmark cleaning and smoothing for Phase 2 Step 2.3.

This service transforms raw per-frame hand landmarks into a cleaner,
analysis-ready signal while preserving frame count and timestamps.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from app.schemas.landmark_schema import (
    FrameLandmarks,
    HandLandmarks,
    LandmarkPoint,
    VideoLandmarksOutput,
)
from app.services.hand_detection_service import process_video_to_landmarks
from app.utils.sequence_utils import smooth_sequence

DEFAULT_SMOOTHING_WINDOW = 3
DEFAULT_MAX_INTERPOLATION_GAP = 2
SWAP_COST_MARGIN = 1e-6
EDGE_BLEND_EPSILON = 1e-9


@dataclass
class LandmarkCleaningSummary:
    total_frames: int
    raw_detected_frames: int
    cleaned_detected_frames: int
    left_gaps_filled: int
    right_gaps_filled: int
    still_missing_left_frames: int
    still_missing_right_frames: int
    handedness_swaps_applied: int


def _clone_point(point: LandmarkPoint) -> LandmarkPoint:
    return LandmarkPoint(
        x=float(point.x),
        y=float(point.y),
        z=float(point.z) if point.z is not None else None,
        confidence=float(point.confidence) if point.confidence is not None else None,
    )


def _clone_hand(hand: Optional[HandLandmarks]) -> Optional[HandLandmarks]:
    if hand is None:
        return None
    return HandLandmarks(landmarks=[_clone_point(point) for point in hand.landmarks])


def _clone_frame(frame: FrameLandmarks) -> FrameLandmarks:
    return FrameLandmarks(
        frame_index=frame.frame_index,
        timestamp_sec=frame.timestamp_sec,
        left_hand=_clone_hand(frame.left_hand),
        right_hand=_clone_hand(frame.right_hand),
    )


def _clone_video_landmarks(video_landmarks: VideoLandmarksOutput) -> VideoLandmarksOutput:
    return VideoLandmarksOutput(
        video_id=video_landmarks.video_id,
        fps=video_landmarks.fps,
        total_frames=video_landmarks.total_frames,
        frames=[_clone_frame(frame) for frame in video_landmarks.frames],
        video_path=video_landmarks.video_path,
        coordinate_system=video_landmarks.coordinate_system,
        vjepa_features=video_landmarks.vjepa_features,
    )


def _wrist_xy(hand: Optional[HandLandmarks]) -> Optional[tuple[float, float]]:
    if hand is None or not hand.landmarks:
        return None
    wrist = hand.landmarks[0]
    return wrist.x, wrist.y


def _distance(point_a: Optional[tuple[float, float]], point_b: Optional[tuple[float, float]]) -> float:
    if point_a is None or point_b is None:
        return 0.0
    dx = point_a[0] - point_b[0]
    dy = point_a[1] - point_b[1]
    return (dx * dx + dy * dy) ** 0.5


def _stabilize_handedness(frames: list[FrameLandmarks]) -> int:
    swaps_applied = 0
    previous_left: Optional[HandLandmarks] = None
    previous_right: Optional[HandLandmarks] = None

    for frame in frames:
        if frame.left_hand is not None and frame.right_hand is not None:
            current_cost = (
                _distance(_wrist_xy(previous_left), _wrist_xy(frame.left_hand))
                + _distance(_wrist_xy(previous_right), _wrist_xy(frame.right_hand))
            )
            swapped_cost = (
                _distance(_wrist_xy(previous_left), _wrist_xy(frame.right_hand))
                + _distance(_wrist_xy(previous_right), _wrist_xy(frame.left_hand))
            )

            if previous_left is not None and previous_right is not None:
                if swapped_cost + SWAP_COST_MARGIN < current_cost:
                    frame.left_hand, frame.right_hand = frame.right_hand, frame.left_hand
                    swaps_applied += 1

        if frame.left_hand is not None:
            previous_left = frame.left_hand
        if frame.right_hand is not None:
            previous_right = frame.right_hand

    return swaps_applied


def _interpolate_between_hands(
    start_hand: HandLandmarks,
    end_hand: HandLandmarks,
    gap_length: int,
) -> list[HandLandmarks]:
    interpolated: list[HandLandmarks] = []

    for missing_offset in range(1, gap_length + 1):
        alpha = missing_offset / (gap_length + 1)
        interpolated_points: list[LandmarkPoint] = []

        for start_point, end_point in zip(start_hand.landmarks, end_hand.landmarks):
            start_z = start_point.z if start_point.z is not None else 0.0
            end_z = end_point.z if end_point.z is not None else 0.0
            interpolated_points.append(
                LandmarkPoint(
                    x=(1 - alpha) * start_point.x + alpha * end_point.x,
                    y=(1 - alpha) * start_point.y + alpha * end_point.y,
                    z=(1 - alpha) * start_z + alpha * end_z,
                )
            )

        interpolated.append(HandLandmarks(landmarks=interpolated_points))

    return interpolated


def _fill_short_gaps_for_hand(
    frames: list[FrameLandmarks],
    hand_side: str,
    max_gap_frames: int,
) -> int:
    gaps_filled = 0
    index = 0

    while index < len(frames):
        current_hand = getattr(frames[index], hand_side)
        if current_hand is not None:
            index += 1
            continue

        gap_start = index
        while index < len(frames) and getattr(frames[index], hand_side) is None:
            index += 1
        gap_end = index - 1
        gap_length = gap_end - gap_start + 1

        prev_index = gap_start - 1
        next_index = index if index < len(frames) else None

        if (
            gap_length <= max_gap_frames
            and prev_index >= 0
            and next_index is not None
            and getattr(frames[prev_index], hand_side) is not None
            and getattr(frames[next_index], hand_side) is not None
        ):
            interpolated_hands = _interpolate_between_hands(
                getattr(frames[prev_index], hand_side),
                getattr(frames[next_index], hand_side),
                gap_length,
            )
            for offset, hand in enumerate(interpolated_hands):
                setattr(frames[gap_start + offset], hand_side, hand)
            gaps_filled += gap_length

    return gaps_filled


def _contiguous_non_null_segments(
    hands: list[Optional[HandLandmarks]],
) -> list[tuple[int, int]]:
    segments: list[tuple[int, int]] = []
    start: Optional[int] = None

    for index, hand in enumerate(hands):
        if hand is not None and start is None:
            start = index
        elif hand is None and start is not None:
            segments.append((start, index - 1))
            start = None

    if start is not None:
        segments.append((start, len(hands) - 1))

    return segments


def _edge_aware_smooth_sequence(values: list[float], window: int) -> list[float]:
    if len(values) <= 1 or window <= 1:
        return values

    base_smoothed = smooth_sequence(values, window=window)
    edge_span = max(1, window - 1)
    last_index = len(values) - 1
    result: list[float] = []

    for index, (raw_value, smooth_value) in enumerate(zip(values, base_smoothed)):
        distance_to_edge = min(index, last_index - index)
        blend = min(1.0, distance_to_edge / edge_span)
        if blend <= EDGE_BLEND_EPSILON:
            result.append(raw_value)
        else:
            result.append((1.0 - blend) * raw_value + blend * smooth_value)

    return result


def _smooth_hand_sequence(
    frames: list[FrameLandmarks],
    hand_side: str,
    smoothing_window: int,
) -> None:
    hands = [getattr(frame, hand_side) for frame in frames]

    for segment_start, segment_end in _contiguous_non_null_segments(hands):
        segment_hands = hands[segment_start : segment_end + 1]
        if len(segment_hands) < 2:
            continue

        for landmark_index in range(21):
            xs = [hand.landmarks[landmark_index].x for hand in segment_hands if hand is not None]
            ys = [hand.landmarks[landmark_index].y for hand in segment_hands if hand is not None]
            zs = [
                hand.landmarks[landmark_index].z if hand.landmarks[landmark_index].z is not None else 0.0
                for hand in segment_hands
                if hand is not None
            ]

            smooth_x = _edge_aware_smooth_sequence(xs, window=smoothing_window)
            smooth_y = _edge_aware_smooth_sequence(ys, window=smoothing_window)
            smooth_z = _edge_aware_smooth_sequence(zs, window=smoothing_window)

            for offset, hand in enumerate(segment_hands):
                hand.landmarks[landmark_index].x = float(smooth_x[offset])
                hand.landmarks[landmark_index].y = float(smooth_y[offset])
                hand.landmarks[landmark_index].z = float(smooth_z[offset])


def clean_video_landmarks(
    video_landmarks: VideoLandmarksOutput,
    *,
    smoothing_window: int = DEFAULT_SMOOTHING_WINDOW,
    max_gap_frames: int = DEFAULT_MAX_INTERPOLATION_GAP,
) -> VideoLandmarksOutput:
    """Clean and smooth a raw landmark sequence while preserving timing."""
    if smoothing_window <= 0:
        raise ValueError("smoothing_window must be greater than zero.")
    if max_gap_frames < 0:
        raise ValueError("max_gap_frames must be zero or greater.")

    cleaned = _clone_video_landmarks(video_landmarks)
    _stabilize_handedness(cleaned.frames)
    _fill_short_gaps_for_hand(cleaned.frames, "left_hand", max_gap_frames)
    _fill_short_gaps_for_hand(cleaned.frames, "right_hand", max_gap_frames)
    _smooth_hand_sequence(cleaned.frames, "left_hand", smoothing_window)
    _smooth_hand_sequence(cleaned.frames, "right_hand", smoothing_window)
    return cleaned


def summarize_cleaning(
    raw_landmarks: VideoLandmarksOutput,
    cleaned_landmarks: VideoLandmarksOutput,
    *,
    max_gap_frames: int = DEFAULT_MAX_INTERPOLATION_GAP,
) -> LandmarkCleaningSummary:
    raw_detected_frames = sum(
        1 for frame in raw_landmarks.frames if frame.left_hand is not None or frame.right_hand is not None
    )
    cleaned_detected_frames = sum(
        1 for frame in cleaned_landmarks.frames if frame.left_hand is not None or frame.right_hand is not None
    )

    left_raw_missing = sum(1 for frame in raw_landmarks.frames if frame.left_hand is None)
    right_raw_missing = sum(1 for frame in raw_landmarks.frames if frame.right_hand is None)
    left_clean_missing = sum(1 for frame in cleaned_landmarks.frames if frame.left_hand is None)
    right_clean_missing = sum(1 for frame in cleaned_landmarks.frames if frame.right_hand is None)

    # Re-run stabilization count deterministically on a clone of raw to report swap decisions.
    raw_clone_for_stats = _clone_video_landmarks(raw_landmarks)
    swaps_applied = _stabilize_handedness(raw_clone_for_stats.frames)
    left_gaps_filled = max(0, left_raw_missing - left_clean_missing)
    right_gaps_filled = max(0, right_raw_missing - right_clean_missing)

    return LandmarkCleaningSummary(
        total_frames=cleaned_landmarks.total_frames,
        raw_detected_frames=raw_detected_frames,
        cleaned_detected_frames=cleaned_detected_frames,
        left_gaps_filled=left_gaps_filled,
        right_gaps_filled=right_gaps_filled,
        still_missing_left_frames=left_clean_missing,
        still_missing_right_frames=right_clean_missing,
        handedness_swaps_applied=swaps_applied,
    )


def process_video_to_clean_landmarks(video_path: str | Path) -> VideoLandmarksOutput:
    """Run preprocessing, hand detection, then landmark cleaning."""
    raw_landmarks = process_video_to_landmarks(video_path)
    return clean_video_landmarks(raw_landmarks)
