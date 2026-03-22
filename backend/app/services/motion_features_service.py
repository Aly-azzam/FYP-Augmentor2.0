"""Motion Features Service — compute kinematics from sequences.

Responsibilities:
- compute velocity per landmark per frame
- compute speed (magnitude of velocity)
- compute acceleration if needed

Owner: Motion Representation Engine (Person 2)
"""

from app.utils.math_utils import euclidean_distance, zero_vector
from app.utils.sequence_utils import compute_dt


class MotionFeaturesService:
    """Compute basic motion features from position and velocity series."""

    def compute_velocity(self, series: list[list[float]], fps: float) -> list[list[float]]:
        """Compute per-frame velocity vectors from a position series."""
        if not series:
            return []

        dt = compute_dt(fps)
        dim = len(series[0]) if series[0] else 3
        velocities: list[list[float]] = [zero_vector(dim)]

        for index in range(1, len(series)):
            previous = series[index - 1]
            current = series[index]

            if len(previous) != len(current):
                dim = max(len(previous), len(current))
                previous = previous + [0.0] * (dim - len(previous))
                current = current + [0.0] * (dim - len(current))

            velocity = [
                (current[axis] - previous[axis]) / dt
                for axis in range(len(current))
            ]
            velocities.append(velocity)

        return velocities

    def compute_speed(self, velocity_vectors: list[list[float]]) -> list[float]:
        """Compute scalar speed from velocity vectors."""
        if not velocity_vectors:
            return []

        return [euclidean_distance(vector, zero_vector(len(vector))) for vector in velocity_vectors]

    def compute_acceleration(
        self,
        velocity_vectors: list[list[float]],
        fps: float,
    ) -> list[list[float]]:
        """Compute per-frame acceleration vectors from velocity vectors."""
        if not velocity_vectors:
            return []

        dt = compute_dt(fps)
        dim = len(velocity_vectors[0]) if velocity_vectors[0] else 3
        accelerations: list[list[float]] = [zero_vector(dim)]

        for index in range(1, len(velocity_vectors)):
            previous = velocity_vectors[index - 1]
            current = velocity_vectors[index]

            if len(previous) != len(current):
                dim = max(len(previous), len(current))
                previous = previous + [0.0] * (dim - len(previous))
                current = current + [0.0] * (dim - len(current))

            acceleration = [
                (current[axis] - previous[axis]) / dt
                for axis in range(len(current))
            ]
            accelerations.append(acceleration)

        return accelerations


async def compute_kinematics(sequence_data: dict, fps: float) -> dict:
    """Compute velocity and speed for each tracked landmark.

    Uses time delta derived from fps. Returns dict with velocity and speed arrays.
    """
    # TODO: implement kinematic computation in normalized coordinates
    return {
        "velocity": {},
        "speed": {},
    }
