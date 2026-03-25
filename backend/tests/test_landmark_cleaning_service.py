from __future__ import annotations

from app.schemas.landmark_schema import (
    FrameLandmarks,
    HandLandmarks,
    LandmarkPoint,
    VideoLandmarksOutput,
)
from app.services.landmark_cleaning_service import (
    clean_video_landmarks,
    summarize_cleaning,
)


def _make_hand(x: float, y: float, z: float = 0.0) -> HandLandmarks:
    return HandLandmarks(
        landmarks=[
            LandmarkPoint(x=x + landmark_index * 0.001, y=y + landmark_index * 0.001, z=z)
            for landmark_index in range(21)
        ]
    )


def _make_output(left_hands, right_hands=None) -> VideoLandmarksOutput:
    if right_hands is None:
        right_hands = [None] * len(left_hands)

    frames = [
        FrameLandmarks(
            frame_index=index,
            timestamp_sec=index / 30.0,
            left_hand=left_hands[index],
            right_hand=right_hands[index],
        )
        for index in range(len(left_hands))
    ]

    return VideoLandmarksOutput(
        video_id="demo",
        fps=30.0,
        total_frames=len(frames),
        frames=frames,
        video_path="demo.mp4",
        coordinate_system="normalized",
    )


def _wrist_x_series(video_landmarks: VideoLandmarksOutput) -> list[float]:
    values = []
    for frame in video_landmarks.frames:
        if frame.left_hand is not None:
            values.append(frame.left_hand.landmarks[0].x)
    return values


def _jitter(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    return sum(abs(values[index] - values[index - 1]) for index in range(1, len(values)))


def _frame_wrist_x(video_landmarks: VideoLandmarksOutput, frame_index: int) -> float:
    hand = video_landmarks.frames[frame_index].left_hand
    assert hand is not None
    return hand.landmarks[0].x


def test_clean_video_landmarks_preserves_structure_and_timing() -> None:
    raw = _make_output([_make_hand(0.2, 0.3), _make_hand(0.25, 0.35), _make_hand(0.3, 0.4)])
    cleaned = clean_video_landmarks(raw)

    assert cleaned.total_frames == raw.total_frames
    assert [frame.frame_index for frame in cleaned.frames] == [frame.frame_index for frame in raw.frames]
    assert [frame.timestamp_sec for frame in cleaned.frames] == [frame.timestamp_sec for frame in raw.frames]
    assert len(cleaned.frames[0].left_hand.landmarks) == 21


def test_clean_video_landmarks_reduces_jitter() -> None:
    raw = _make_output(
        [
            _make_hand(0.10, 0.20),
            _make_hand(0.16, 0.21),
            _make_hand(0.11, 0.22),
            _make_hand(0.17, 0.23),
            _make_hand(0.12, 0.24),
        ]
    )

    cleaned = clean_video_landmarks(raw, smoothing_window=3)

    raw_series = _wrist_x_series(raw)
    cleaned_series = _wrist_x_series(cleaned)

    # Preserve the boundaries exactly for the default edge-aware smoothing behavior.
    assert _frame_wrist_x(cleaned, 0) == _frame_wrist_x(raw, 0)
    assert _frame_wrist_x(cleaned, len(raw.frames) - 1) == _frame_wrist_x(raw, len(raw.frames) - 1)

    # Middle frames should still benefit from smoothing.
    assert _jitter(cleaned_series[1:-1]) < _jitter(raw_series[1:-1])
    assert abs(_frame_wrist_x(cleaned, 0) - _frame_wrist_x(raw, 0)) < 0.02
    assert abs(_frame_wrist_x(cleaned, len(raw.frames) - 1) - _frame_wrist_x(raw, len(raw.frames) - 1)) < 0.02


def test_clean_video_landmarks_fills_short_gaps_but_keeps_long_ones() -> None:
    raw = _make_output(
        [
            _make_hand(0.10, 0.20),
            None,
            None,
            _make_hand(0.16, 0.26),
            None,
            None,
            None,
            _make_hand(0.25, 0.35),
        ]
    )

    cleaned = clean_video_landmarks(raw, max_gap_frames=2)

    assert cleaned.frames[1].left_hand is not None
    assert cleaned.frames[2].left_hand is not None
    assert cleaned.frames[4].left_hand is None
    assert cleaned.frames[5].left_hand is None
    assert cleaned.frames[6].left_hand is None

    summary = summarize_cleaning(raw, cleaned)
    assert summary.left_gaps_filled == 2
    assert summary.still_missing_left_frames == 3


def test_clean_video_landmarks_stabilizes_simple_handedness_swap() -> None:
    raw = _make_output(
        left_hands=[_make_hand(0.2, 0.2), _make_hand(0.8, 0.8), _make_hand(0.22, 0.22)],
        right_hands=[_make_hand(0.8, 0.8), _make_hand(0.2, 0.2), _make_hand(0.82, 0.82)],
    )

    cleaned = clean_video_landmarks(raw)

    assert cleaned.frames[1].left_hand is not None
    assert cleaned.frames[1].right_hand is not None
    assert cleaned.frames[1].left_hand.landmarks[0].x < cleaned.frames[1].right_hand.landmarks[0].x


def test_clean_video_landmarks_preserves_edges_better_than_middle() -> None:
    raw = _make_output(
        [
            _make_hand(0.20, 0.20),
            _make_hand(0.60, 0.55),
            _make_hand(0.59, 0.54),
            _make_hand(0.58, 0.53),
            _make_hand(0.57, 0.52),
        ]
    )

    cleaned = clean_video_landmarks(raw, smoothing_window=3)

    first_delta = abs(_frame_wrist_x(cleaned, 0) - _frame_wrist_x(raw, 0))
    middle_delta = abs(_frame_wrist_x(cleaned, 1) - _frame_wrist_x(raw, 1))
    last_delta = abs(_frame_wrist_x(cleaned, 4) - _frame_wrist_x(raw, 4))

    assert first_delta < middle_delta
    assert last_delta <= middle_delta
    assert first_delta < 0.02
    assert last_delta < 0.02
