"""Alignment Preparation Service — prepare data for expert/learner comparison.

Responsibilities:
- temporal alignment between expert and learner sequences
- DTW or simple frame-matching preparation
- output aligned pairs for evaluation

Owner: Motion Representation Engine (Person 2)
"""

from app.schemas.motion_schema import MotionRepresentationOutput
from app.utils.math_utils import zero_vector


class AlignmentPrepService:
    """Prepare normalized per-frame vectors for future alignment algorithms."""

    def prepare_alignment_vectors(
        self,
        sequence: MotionRepresentationOutput,
    ) -> dict:
        """Flatten normalized landmark positions into one vector per frame.

        Landmark order is preserved using `sequence.tracked_landmarks`.
        Only normalized positions are included. Velocity, speed, and
        acceleration are intentionally excluded at this step.
        """
        alignment_vectors: list[list[float]] = []

        for frame in sequence.frames:
            frame_vector: list[float] = []

            for landmark_name in sequence.tracked_landmarks:
                position = frame.positions.get(landmark_name)
                if not position:
                    position = zero_vector(3)

                if len(position) < 3:
                    position = position + [0.0] * (3 - len(position))

                frame_vector.extend(position[:3])

            alignment_vectors.append(frame_vector)

        return {
            "video_id": sequence.video_id,
            "fps": sequence.fps,
            "sequence_length": sequence.sequence_length,
            "alignment_vectors": alignment_vectors,
        }


async def prepare_alignment(expert_motion: dict, learner_motion: dict) -> dict:
    """Prepare aligned frame pairs for comparison.

    Returns dict with aligned_pairs and alignment metadata.
    """
    # TODO: implement DTW or simple temporal alignment
    return {
        "aligned_pairs": [],
        "alignment_method": "none",
        "expert_length": 0,
        "learner_length": 0,
    }
