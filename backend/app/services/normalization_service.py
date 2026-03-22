"""Normalization Service — coordinate and temporal normalization.

Responsibilities:
- ensure all data is in canonical normalized coordinate space
- normalize tool bboxes from pixel to normalized if needed
- validate coordinate conventions

Owner: Motion Representation Engine (Person 2)
"""

from app.schemas.motion_schema import MotionRepresentationOutput
from app.utils.math_utils import euclidean_distance


class NormalizationService:
    """Normalize motion positions frame-by-frame for comparison."""

    reference_landmark = "wrist"
    scale_landmark = "index_tip"

    def normalize_positions(
        self,
        sequence: MotionRepresentationOutput,
    ) -> MotionRepresentationOutput:
        """Normalize frame positions using wrist centering and wrist-index scaling.

        For each frame:
        1. Use wrist as the reference point and subtract it from all positions.
        2. Use the wrist-to-index_tip distance as the scale factor.
        3. If the scale factor is zero, keep the frame unchanged.

        Velocity, speed, and acceleration are intentionally left unchanged in Step 5.
        """
        normalized_sequence = sequence.model_copy(deep=True)

        for frame in normalized_sequence.frames:
            wrist = frame.positions.get(self.reference_landmark)
            index_tip = frame.positions.get(self.scale_landmark)

            if not wrist or not index_tip:
                continue

            scale_factor = euclidean_distance(wrist, index_tip)
            if scale_factor == 0.0:
                continue

            normalized_positions: dict[str, list[float]] = {}
            normalized_trajectory_points: dict[str, list[float]] = {}

            for landmark_name, vector in frame.positions.items():
                centered_vector = [
                    (vector[axis] - wrist[axis]) / scale_factor
                    for axis in range(len(vector))
                ]
                normalized_positions[landmark_name] = centered_vector
                normalized_trajectory_points[landmark_name] = centered_vector[:2]

            frame.positions = normalized_positions
            frame.trajectory_point = normalized_trajectory_points

        return normalized_sequence


async def normalize_coordinates(data: dict, source_format: str = "normalized") -> dict:
    """Normalize data to canonical coordinate convention.

    For MVP, data from MediaPipe is already normalized. This service validates
    and transforms tool detections if they arrive in pixel format.
    """
    # TODO: implement pixel-to-normalized transform for tool detections
    return data
