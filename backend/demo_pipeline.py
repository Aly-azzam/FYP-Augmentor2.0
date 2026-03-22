from pathlib import Path

from app.services.alignment_prep_service import AlignmentPrepService
from app.services.motion_representation_service import MotionRepresentationService
from app.services.normalization_service import NormalizationService
from app.services.scoring_service import ScoringService
from app.services.sequence_service import SequenceService
from app.utils.dtw_utils import compute_dtw


SCENARIOS = [
    (
        "GOOD MATCH",
        Path("storage/raw/learner_landmarks_good.json"),
        Path("storage/raw/expert_landmarks_good.json"),
    ),
    (
        "MEDIUM MATCH",
        Path("storage/raw/learner_landmarks_medium.json"),
        Path("storage/raw/expert_landmarks_medium.json"),
    ),
    (
        "POOR MATCH",
        Path("storage/raw/learner_landmarks_poor.json"),
        Path("storage/raw/expert_landmarks_poor.json"),
    ),
]


def get_feedback(score: float) -> str:
    """Return simple demo feedback based on score bands."""
    if score >= 85:
        return "Excellent alignment with expert motion."
    if score >= 55:
        return "Some deviations detected in trajectory and positioning."
    return "Significant differences from expert motion."


def process_landmark_file(path: Path) -> dict:
    """Run the motion pipeline for a single landmark JSON file."""
    sequence_service = SequenceService()
    motion_representation_service = MotionRepresentationService()
    normalization_service = NormalizationService()
    alignment_prep_service = AlignmentPrepService()

    landmark_data = sequence_service.load_landmark_sequence(path)
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

    return {
        "sequence": sequence,
        "motion_representation": motion_representation,
        "normalized_representation": normalized_representation,
        "alignment_payload": alignment_payload,
    }


def main() -> None:
    print("=== AugMentor Demo ===")
    print()

    scoring_service = ScoringService()

    for scenario_name, learner_path, expert_path in SCENARIOS:
        learner_result = process_landmark_file(learner_path)
        expert_result = process_landmark_file(expert_path)

        dtw_result = compute_dtw(
            learner_result["alignment_payload"]["alignment_vectors"],
            expert_result["alignment_payload"]["alignment_vectors"],
        )
        score_result = scoring_service.compute_score(dtw_result["normalized_distance"])

        print(f"[{scenario_name}]")
        print(f"Learner frames: {learner_result['sequence'].sequence_length}")
        print(f"Expert frames: {expert_result['sequence'].sequence_length}")
        print()
        print(f"DTW distance: {dtw_result['distance']:.6f}")
        print(f"Normalized distance: {dtw_result['normalized_distance']:.6f}")
        print()
        print(f"Score: {round(score_result['score'])}")
        print(f"Feedback: {get_feedback(score_result['score'])}")
        print(f"Final Score: {score_result['score']:.2f} / 100")
        print()


if __name__ == "__main__":
    main()
