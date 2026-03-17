from pathlib import Path

from app.services.motion_representation_service import MotionRepresentationService
from app.services.normalization_service import NormalizationService
from app.services.sequence_service import SequenceService


def main() -> None:
    sequence_service = SequenceService()
    motion_representation_service = MotionRepresentationService()
    normalization_service = NormalizationService()

    landmark_data = sequence_service.load_landmark_sequence(
        Path("storage/raw/fake_landmarks.json")
    )
    sequence = sequence_service.build_motion_sequence(landmark_data)
    motion_representation = motion_representation_service.build_motion_representation(
        sequence
    )
    normalized_representation = normalization_service.normalize_positions(
        motion_representation
    )

    print("Normalization test passed.")
    print(f"first frame before normalization: {motion_representation.frames[0].model_dump()}")
    print(f"first frame after normalization: {normalized_representation.frames[0].model_dump()}")


if __name__ == "__main__":
    main()
