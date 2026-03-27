"""Manual smoke test for Phase 2 Step 2.4 motion representation."""

from __future__ import annotations

import argparse
from pathlib import Path

from app.core.config import settings
from app.services.landmark_cleaning_service import process_video_to_clean_landmarks
from app.services.motion_representation_service import process_video_to_motion_representation


def _default_video_path() -> Path:
    return settings.STORAGE_ROOT / "expert" / "pottery.mp4"


def main() -> None:
    parser = argparse.ArgumentParser(description="Smoke test motion representation features.")
    parser.add_argument("--video-path", default=str(_default_video_path()))
    args = parser.parse_args()

    cleaned_landmarks = process_video_to_clean_landmarks(args.video_path)
    motion_output = process_video_to_motion_representation(args.video_path)

    sample_frame = next(
        frame for frame in motion_output.frames
        if frame.left_hand_features.present or frame.right_hand_features.present
    )
    sample_hand = (
        sample_frame.left_hand_features
        if sample_frame.left_hand_features.present
        else sample_frame.right_hand_features
    )

    print("Motion representation succeeded.")
    print(f"total_frames={motion_output.total_frames}")
    print(
        "timestamps_match_input="
        f"{[frame.timestamp_sec for frame in motion_output.frames[:10]] == [frame.timestamp_sec for frame in cleaned_landmarks.frames[:10]]}"
    )
    print(f"sample_frame_index={sample_frame.frame_index}")
    print(f"sample_timestamp_sec={sample_frame.timestamp_sec:.3f}")
    print(f"sample_hand_present={sample_hand.present}")
    print(f"sample_wrist_position={sample_hand.wrist_position}")
    print(f"sample_palm_center={sample_hand.palm_center}")
    print(f"sample_hand_scale={sample_hand.hand_scale:.4f}")
    print(f"sample_relative_distances={sample_hand.relative_distances}")
    print(f"sample_joint_angles={sample_hand.joint_angles}")
    print(f"sample_hand_openness={sample_hand.hand_openness:.4f}")
    print(f"sample_pinch_distance={sample_hand.pinch_distance:.4f}")
    print(f"sample_finger_spread={sample_hand.finger_spread:.4f}")
    print(f"sample_wrist_velocity={sample_hand.wrist_velocity}")
    print(f"sample_palm_velocity={sample_hand.palm_velocity}")
    print(f"sample_wrist_acceleration={sample_hand.wrist_acceleration}")
    print(f"sample_palm_acceleration={sample_hand.palm_acceleration}")
    print(f"hand_vector_length={len(sample_hand.flattened_feature_vector)}")
    print(f"frame_vector_length={len(sample_frame.flattened_feature_vector)}")


if __name__ == "__main__":
    main()
