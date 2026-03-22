import sys
from pathlib import Path

import pytest


# Ensure `app` imports resolve whether pytest is run from `backend/` or repo root.
BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))


@pytest.fixture
def demo_perception_output():
    from app.schemas.landmark_schema import PerceptionOutput

    return PerceptionOutput(
        video_id="demo",
        fps=30.0,
        total_frames=0,
        frames=[],
        vjepa_features=None,
    )


@pytest.fixture
def demo_metrics():
    return {
        "angle_deviation": 10.0,
        "trajectory_deviation": 12.0,
        "velocity_difference": 8.0,
        "tool_alignment_deviation": 6.0,
    }


@pytest.fixture
def demo_normalization_payload():
    return {
        "video_id": "demo",
        "frames": [{"frame_index": 0, "positions": {}}],
    }
