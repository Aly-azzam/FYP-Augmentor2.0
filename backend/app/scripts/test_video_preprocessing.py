"""Manual smoke test for Phase 2 Step 2.1 video standardization."""

from __future__ import annotations

import argparse
from pathlib import Path

from app.core.config import settings
from app.services.video_preprocessing_service import prepare_videos_for_analysis


def _default_expert_path() -> Path:
    return settings.STORAGE_ROOT / "expert" / "pottery.mp4"


def _default_learner_path(expert_path: Path) -> Path:
    learner_root = settings.STORAGE_ROOT / settings.UPLOAD_DIR / settings.LEARNER_UPLOAD_SUBDIR
    learner_videos = sorted(
        path for path in learner_root.rglob("*") if path.is_file() and path.suffix.lower() in {".mp4", ".mov", ".avi", ".webm"}
    ) if learner_root.exists() else []

    return learner_videos[-1] if learner_videos else expert_path


def _print_video_summary(label: str, output) -> None:
    first_five_timestamps = [round(frame.timestamp_sec, 3) for frame in output.frames[:5]]

    print(f"{label}:")
    print(f"  video_path={output.video_path}")
    print(f"  source_fps={output.source_fps}")
    print(f"  normalized_fps={output.fps}")
    print(f"  source_total_frames={output.source_total_frames}")
    print(f"  normalized_total_frames={output.total_frames}")
    print(f"  first_5_timestamps={first_five_timestamps}")
    print(f"  output_resolution={output.width}x{output.height}")
    print(f"  duration_seconds={output.duration_seconds:.3f}")

    sample_frame = output.frames[0]
    print("  sample_frame:")
    print(f"    frame_index={sample_frame.frame_index}")
    print(f"    source_frame_index={sample_frame.source_frame_index}")
    print(f"    timestamp_sec={sample_frame.timestamp_sec:.3f}")
    print(f"    frame_rgb_shape={sample_frame.frame_rgb.shape}")


def main() -> None:
    default_expert = _default_expert_path()
    default_learner = _default_learner_path(default_expert)

    parser = argparse.ArgumentParser(description="Smoke test the video preprocessing service.")
    parser.add_argument("--expert-path", default=str(default_expert))
    parser.add_argument("--learner-path", default=str(default_learner))
    args = parser.parse_args()

    prepared = prepare_videos_for_analysis(
        expert_video_path=args.expert_path,
        learner_video_path=args.learner_path,
    )

    print("Video preprocessing succeeded.")
    print(
        f"target_fps={prepared.target_fps}, "
        f"target_resolution={prepared.target_width}x{prepared.target_height}"
    )
    _print_video_summary("expert_video", prepared.expert_video)
    _print_video_summary("learner_video", prepared.learner_video)


if __name__ == "__main__":
    main()
