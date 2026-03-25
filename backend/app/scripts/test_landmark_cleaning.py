"""Manual smoke test for Phase 2 Step 2.3 landmark cleaning."""

from __future__ import annotations

import argparse
from pathlib import Path

from app.core.config import settings
from app.services.hand_detection_service import process_video_to_landmarks
from app.services.landmark_cleaning_service import (
    clean_video_landmarks,
    summarize_cleaning,
)


def _default_video_path() -> Path:
    return settings.STORAGE_ROOT / "expert" / "pottery.mp4"


def _first_detected_frame_index(frames) -> int:
    for index, frame in enumerate(frames):
        if frame.left_hand is not None or frame.right_hand is not None:
            return index
    return 0


def _wrist_series(frames, start: int, count: int = 5) -> list[tuple[int, float, float, float]]:
    series: list[tuple[int, float, float, float]] = []

    for frame in frames[start : start + count]:
        hand = frame.left_hand or frame.right_hand
        if hand is None:
            series.append((frame.frame_index, frame.timestamp_sec, float("nan"), float("nan")))
            continue

        wrist = hand.landmarks[0]
        series.append((frame.frame_index, frame.timestamp_sec, wrist.x, wrist.y))

    return series


def _print_series(label: str, raw_frames, cleaned_frames, start: int, count: int = 10) -> None:
    raw_series = _wrist_series(raw_frames, start, count)
    cleaned_series = _wrist_series(cleaned_frames, start, count)

    print(label)
    for raw_item, cleaned_item in zip(raw_series, cleaned_series):
        raw_x, raw_y = raw_item[2], raw_item[3]
        clean_x, clean_y = cleaned_item[2], cleaned_item[3]
        print(
            "  "
            f"frame={raw_item[0]}, timestamp={raw_item[1]:.3f}, "
            f"raw=({raw_x:.4f}, {raw_y:.4f}), "
            f"cleaned=({clean_x:.4f}, {clean_y:.4f}), "
            f"delta=({clean_x - raw_x:+.4f}, {clean_y - raw_y:+.4f})"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Smoke test landmark cleaning and smoothing.")
    parser.add_argument("--video-path", default=str(_default_video_path()))
    args = parser.parse_args()

    raw_landmarks = process_video_to_landmarks(args.video_path)
    cleaned_landmarks = clean_video_landmarks(raw_landmarks)
    summary = summarize_cleaning(raw_landmarks, cleaned_landmarks)

    first_index = _first_detected_frame_index(raw_landmarks.frames)
    middle_index = max(0, (len(raw_landmarks.frames) // 2) - 5)
    last_index = max(0, len(raw_landmarks.frames) - 10)

    print("Landmark cleaning succeeded.")
    print(f"total_frames={summary.total_frames}")
    print(f"raw_detected_frames={summary.raw_detected_frames}")
    print(f"cleaned_detected_frames={summary.cleaned_detected_frames}")
    print(f"left_gaps_filled={summary.left_gaps_filled}")
    print(f"right_gaps_filled={summary.right_gaps_filled}")
    print(f"still_missing_left_frames={summary.still_missing_left_frames}")
    print(f"still_missing_right_frames={summary.still_missing_right_frames}")
    print(f"handedness_swaps_applied={summary.handedness_swaps_applied}")
    _print_series("first_10_wrist_series:", raw_landmarks.frames, cleaned_landmarks.frames, first_index, count=10)
    _print_series("middle_10_wrist_series:", raw_landmarks.frames, cleaned_landmarks.frames, middle_index, count=10)
    _print_series("last_10_wrist_series:", raw_landmarks.frames, cleaned_landmarks.frames, last_index, count=10)


if __name__ == "__main__":
    main()
