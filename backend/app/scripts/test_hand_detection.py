"""Manual smoke test for Phase 2 Step 2.2 hand detection."""

from __future__ import annotations

import argparse
from pathlib import Path

from app.core.config import settings
from app.services.hand_detection_service import extract_hand_landmarks
from app.services.video_preprocessing_service import prepare_videos_for_analysis


def _default_expert_path() -> Path:
    return settings.STORAGE_ROOT / "expert" / "pottery.mp4"


def _default_learner_path(expert_path: Path) -> Path:
    learner_root = settings.STORAGE_ROOT / settings.UPLOAD_DIR / settings.LEARNER_UPLOAD_SUBDIR
    learner_videos = sorted(
        path
        for path in learner_root.rglob("*")
        if path.is_file() and path.suffix.lower() in {".mp4", ".mov", ".avi", ".webm"}
    ) if learner_root.exists() else []

    return learner_videos[-1] if learner_videos else expert_path


def _detected_frames_count(output) -> int:
    return sum(
        1
        for frame in output.frames
        if frame.left_hand is not None or frame.right_hand is not None
    )


def _print_video_summary(label: str, output) -> None:
    print(f"{label}:")
    print(f"  total_frames={output.total_frames}")
    print(f"  detected_hand_frames={_detected_frames_count(output)}")

    sample_frame = next(
        (frame for frame in output.frames if frame.left_hand is not None or frame.right_hand is not None),
        output.frames[0],
    )
    print("  sample_frame:")
    print(f"    frame_index={sample_frame.frame_index}")
    print(f"    timestamp_sec={sample_frame.timestamp_sec:.3f}")

    if sample_frame.left_hand is not None:
        print(f"    left_hand_landmarks={len(sample_frame.left_hand.landmarks)}")
        first_point = sample_frame.left_hand.landmarks[0]
        print(f"    left_hand_first_point=({first_point.x:.4f}, {first_point.y:.4f}, {first_point.z:.4f})")
    else:
        print("    left_hand_landmarks=None")

    if sample_frame.right_hand is not None:
        print(f"    right_hand_landmarks={len(sample_frame.right_hand.landmarks)}")
        first_point = sample_frame.right_hand.landmarks[0]
        print(f"    right_hand_first_point=({first_point.x:.4f}, {first_point.y:.4f}, {first_point.z:.4f})")
    else:
        print("    right_hand_landmarks=None")


def main() -> None:
    default_expert = _default_expert_path()
    default_learner = _default_learner_path(default_expert)

    parser = argparse.ArgumentParser(description="Smoke test the hand detection service.")
    parser.add_argument("--expert-path", default=str(default_expert))
    parser.add_argument("--learner-path", default=str(default_learner))
    args = parser.parse_args()

    prepared = prepare_videos_for_analysis(
        expert_video_path=args.expert_path,
        learner_video_path=args.learner_path,
    )
    expert_output = extract_hand_landmarks(prepared.expert_video)
    learner_output = extract_hand_landmarks(prepared.learner_video)

    print("Hand detection succeeded.")
    _print_video_summary("expert_video", expert_output)
    _print_video_summary("learner_video", learner_output)


if __name__ == "__main__":
    main()
