"""Trajectory Service — extract trajectory paths from sequences.

Responsibilities:
- compute per-landmark trajectory paths
- smooth trajectories if needed

Owner: Motion Representation Engine (Person 2)
"""

from app.schemas.motion_schema import MotionSequenceOutput
from app.utils.math_utils import euclidean_distance


class TrajectoryService:
    """Build simple 2D trajectories from cleaned motion sequences."""

    def build_trajectory(
        self,
        sequence: MotionSequenceOutput,
        landmark_name: str,
    ) -> list[list[float]]:
        """Extract [x, y] trajectory points for a single landmark."""
        if landmark_name not in sequence.tracked_landmarks:
            raise ValueError(f"Untracked landmark requested: {landmark_name}")

        trajectory: list[list[float]] = []
        for frame in sequence.frames:
            vector = frame.positions.get(landmark_name, [0.0, 0.0, 0.0])
            x = float(vector[0]) if len(vector) > 0 else 0.0
            y = float(vector[1]) if len(vector) > 1 else 0.0
            trajectory.append([x, y])

        return trajectory

    def compute_trajectory_length(self, points: list[list[float]]) -> float:
        """Compute the total distance traveled along a trajectory."""
        if len(points) < 2:
            return 0.0

        total_distance = 0.0
        for index in range(1, len(points)):
            total_distance += euclidean_distance(points[index - 1], points[index])
        return total_distance


async def extract_trajectories(sequence_data: dict) -> list[dict]:
    """Extract trajectory points for each tracked landmark.

    Returns list of TrajectoryPoint-compatible dicts.
    """
    # TODO: implement trajectory extraction
    return []
