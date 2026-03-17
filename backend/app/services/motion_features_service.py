"""Motion Features Service — compute kinematics from sequences.

Responsibilities:
- compute velocity per landmark per frame
- compute speed (magnitude of velocity)
- compute acceleration if needed

Owner: Motion Representation Engine (Person 2)
"""


async def compute_kinematics(sequence_data: dict, fps: float) -> dict:
    """Compute velocity and speed for each tracked landmark.

    Uses time delta derived from fps. Returns dict with velocity and speed arrays.
    """
    # TODO: implement kinematic computation in normalized coordinates
    return {
        "velocity": {},
        "speed": {},
    }
