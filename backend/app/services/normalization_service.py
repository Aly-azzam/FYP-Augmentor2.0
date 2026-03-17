"""Normalization Service — coordinate and temporal normalization.

Responsibilities:
- ensure all data is in canonical normalized coordinate space
- normalize tool bboxes from pixel to normalized if needed
- validate coordinate conventions

Owner: Motion Representation Engine (Person 2)
"""


async def normalize_coordinates(data: dict, source_format: str = "normalized") -> dict:
    """Normalize data to canonical coordinate convention.

    For MVP, data from MediaPipe is already normalized. This service validates
    and transforms tool detections if they arrive in pixel format.
    """
    # TODO: implement pixel-to-normalized transform for tool detections
    return data
