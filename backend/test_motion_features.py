from pathlib import Path

from app.services.motion_features_service import MotionFeaturesService
from app.services.sequence_service import SequenceService
from app.services.trajectory_service import TrajectoryService


def main() -> None:
    sequence_service = SequenceService()
    motion_features_service = MotionFeaturesService()
    trajectory_service = TrajectoryService()

    landmark_data = sequence_service.load_landmark_sequence(
        Path("storage/raw/fake_landmarks.json")
    )
    sequence = sequence_service.build_motion_sequence(landmark_data)
    wrist_series = sequence_service.extract_landmark_series(sequence, "wrist")

    velocity = motion_features_service.compute_velocity(wrist_series, sequence.fps)
    speed = motion_features_service.compute_speed(velocity)
    acceleration = motion_features_service.compute_acceleration(velocity, sequence.fps)

    trajectory = trajectory_service.build_trajectory(sequence, "wrist")
    trajectory_length = trajectory_service.compute_trajectory_length(trajectory)

    print("Motion feature test passed.")
    print(f"first velocity: {velocity[0]}")
    print(f"first speed: {speed[0]}")
    print(f"first acceleration: {acceleration[0]}")
    print(f"trajectory length: {trajectory_length:.6f}")


if __name__ == "__main__":
    main()
