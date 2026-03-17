"""V-JEPA Service — placeholder for V-JEPA feature extraction.

This is a planned future integration. For MVP, this module returns None/empty
features and is treated as optional in the pipeline.

Owner: Perception Engine (Person 1)
"""

from pathlib import Path
from typing import Optional


async def extract_vjepa_features(video_path: Path) -> Optional[dict]:
    """Extract V-JEPA features from video.

    Returns None in MVP — this is a placeholder.
    """
    # TODO: integrate V-JEPA model when ready
    return None
