import asyncio

from app.schemas.landmark_schema import PerceptionOutput
from app.services.motion_representation_service import run_motion_representation


def test_run_motion_representation_returns_stub_outputs() -> None:
    perception_output = PerceptionOutput(
        video_id="demo",
        fps=30.0,
        total_frames=0,
        frames=[],
        vjepa_features=None,
    )

    result = asyncio.run(
        run_motion_representation(
            expert_perception=perception_output,
            learner_perception=perception_output,
        )
    )

    assert set(result.keys()) == {"expert_motion", "learner_motion", "alignment"}
    assert result["alignment"] == {}

    expert_motion = result["expert_motion"]
    learner_motion = result["learner_motion"]

    assert expert_motion.video_id == "stub"
    assert learner_motion.video_id == "stub"
    assert expert_motion.sequence_length == 0
    assert learner_motion.sequence_length == 0
    assert expert_motion.frames == []
    assert learner_motion.frames == []
