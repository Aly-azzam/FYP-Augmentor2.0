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
from app.schemas.motion_schema import (
    EnrichedMotionFrame,
    MotionOutput,
    MotionRepresentationOutput,
    MotionSequenceOutput,
)
from app.services.motion_features_service import MotionFeaturesService
from app.services.sequence_service import SequenceService
from app.services.trajectory_service import TrajectoryService


class MotionRepresentationService:
    """Assemble enriched per-frame motion information from a clean sequence."""

    def __init__(self) -> None:
        self.sequence_service = SequenceService()
        self.trajectory_service = TrajectoryService()
        self.motion_features_service = MotionFeaturesService()

    def build_motion_representation(
        self,
        sequence: MotionSequenceOutput,
    ) -> MotionRepresentationOutput:
        """Build the enriched motion representation for all tracked landmarks."""
        frame_count = len(sequence.frames)

        velocity_by_landmark: dict[str, list[list[float]]] = {}
        speed_by_landmark: dict[str, list[float]] = {}
        acceleration_by_landmark: dict[str, list[list[float]]] = {}
        trajectory_by_landmark: dict[str, list[list[float]]] = {}

        for landmark_name in sequence.tracked_landmarks:
            series = self.sequence_service.extract_landmark_series(sequence, landmark_name)
            trajectory = self.trajectory_service.build_trajectory(sequence, landmark_name)
            velocity = self.motion_features_service.compute_velocity(series, sequence.fps)
            speed = self.motion_features_service.compute_speed(velocity)
            acceleration = self.motion_features_service.compute_acceleration(
                velocity,
                sequence.fps,
            )

            velocity_by_landmark[landmark_name] = velocity
            speed_by_landmark[landmark_name] = speed
            acceleration_by_landmark[landmark_name] = acceleration
            trajectory_by_landmark[landmark_name] = trajectory

        enriched_frames: list[EnrichedMotionFrame] = []
        for frame_index, frame in enumerate(sequence.frames):
            enriched_frames.append(
                EnrichedMotionFrame(
                    frame_index=frame.frame_index,
                    timestamp=frame.timestamp,
                    positions=frame.positions,
                    velocity={
                        landmark_name: velocity_by_landmark[landmark_name][frame_index]
                        for landmark_name in sequence.tracked_landmarks
                    },
                    speed={
                        landmark_name: speed_by_landmark[landmark_name][frame_index]
                        for landmark_name in sequence.tracked_landmarks
                    },
                    acceleration={
                        landmark_name: acceleration_by_landmark[landmark_name][frame_index]
                        for landmark_name in sequence.tracked_landmarks
                    },
                    trajectory_point={
                        landmark_name: trajectory_by_landmark[landmark_name][frame_index]
                        for landmark_name in sequence.tracked_landmarks
                    },
                )
            )

        return MotionRepresentationOutput(
            video_id=sequence.video_id,
            fps=sequence.fps,
            sequence_length=sequence.sequence_length,
            tracked_landmarks=sequence.tracked_landmarks,
            frames=enriched_frames,
        )


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
        frames=[],
    )
    return {
        "expert_motion": stub_output,
        "learner_motion": stub_output,
        "alignment": {},
    }
