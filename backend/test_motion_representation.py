from pathlib import Path

from app.services.motion_representation_service import MotionRepresentationService
from app.services.sequence_service import SequenceService


def main() -> None:
    sequence_service = SequenceService()
    motion_representation_service = MotionRepresentationService()

    landmark_data = sequence_service.load_landmark_sequence(
        Path("storage/raw/fake_landmarks.json")
    )
    sequence = sequence_service.build_motion_sequence(landmark_data)
    motion_representation = motion_representation_service.build_motion_representation(
        sequence
    )

    assert motion_representation.video_id == "learner_01"
    assert motion_representation.sequence_length == 4
    assert motion_representation.tracked_landmarks == [
        "wrist",
        "index_tip",
        "thumb_tip",
    ]
    assert motion_representation.frames[0].velocity["wrist"] == [0.0, 0.0, 0.0]
    assert motion_representation.frames[0].speed["wrist"] == 0.0
    assert motion_representation.frames[0].acceleration["wrist"] == [0.0, 0.0, 0.0]
    assert motion_representation.frames[0].trajectory_point["wrist"] == [0.41, 0.76]

    print("Motion representation test passed.")
    print(f"video_id: {motion_representation.video_id}")
    print(f"sequence_length: {motion_representation.sequence_length}")
    print(f"tracked_landmarks: {motion_representation.tracked_landmarks}")
    print(
        "first enriched frame: "
        f"{motion_representation.frames[0].model_dump()}"
    )


if __name__ == "__main__":
    main()
