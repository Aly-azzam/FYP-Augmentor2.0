"""Motion Representation Service — orchestrates full motion stage.

Combines:
- sequence building
- trajectory extraction
- kinematic computation
- normalization
- alignment preparation

Produces: MotionOutput contract

Owner: Motion Representation Engine (Person 2)
"""

from app.schemas.landmark_schema import PerceptionOutput
from app.schemas.motion_schema import MotionOutput


async def run_motion_representation(
    expert_perception: PerceptionOutput,
    learner_perception: PerceptionOutput,
) -> dict:
    """Run full motion representation pipeline.

    Steps:
    1. Build sequences from landmarks
    2. Compute kinematics (velocity, speed)
    3. Extract trajectories
    4. Normalize coordinates
    5. Prepare alignment between expert and learner

    Returns dict with expert_motion, learner_motion, and alignment data.
    """
    # TODO: wire real sub-services
    stub_output = MotionOutput(
        video_id="stub",
        fps=30.0,
        sequence_length=0,
        tracked_landmarks=[],
        tracked_tools=[],
        frames=[],
        trajectories=[],
        tool_states=[],
    )
    return {
        "expert_motion": stub_output,
        "learner_motion": stub_output,
        "alignment": {},
    }
