import asyncio

from app.schemas.landmark_schema import PerceptionOutput
from app.services.sequence_service import build_sequences


def test_build_sequences_returns_expected_stub_shape() -> None:
    perception_output = PerceptionOutput(
        video_id="demo",
        fps=30.0,
        total_frames=0,
        frames=[],
        vjepa_features=None,
    )

    result = asyncio.run(build_sequences(perception_output))

    assert result == {
        "tracked_landmarks": [],
        "frames": [],
        "sequence_length": 0,
    }
