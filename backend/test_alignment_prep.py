from pathlib import Path

from app.services.alignment_prep_service import AlignmentPrepService
from app.services.motion_representation_service import MotionRepresentationService
from app.services.normalization_service import NormalizationService
from app.services.sequence_service import SequenceService


def main() -> None:
    sequence_service = SequenceService()
    motion_representation_service = MotionRepresentationService()
    normalization_service = NormalizationService()
    alignment_prep_service = AlignmentPrepService()

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
    alignment_payload = alignment_prep_service.prepare_alignment_vectors(
        normalized_representation
    )

    assert alignment_payload["sequence_length"] == 4
    assert len(alignment_payload["alignment_vectors"]) == 4
    assert len(alignment_payload["alignment_vectors"][0]) == 9

    print("Alignment prep test passed.")
    print(f"sequence_length: {alignment_payload['sequence_length']}")
    print(f"length of one vector: {len(alignment_payload['alignment_vectors'][0])}")
    print(f"first alignment vector: {alignment_payload['alignment_vectors'][0]}")


if __name__ == "__main__":
    main()
