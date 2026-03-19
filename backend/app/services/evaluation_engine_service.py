"""Evaluation Engine: internal single-video evaluation pipeline.

This module orchestrates the internal evaluation flow for ONE video:
`video -> landmarks -> features -> score`.

It runs perception, derives angle/trajectory/velocity features, computes an
internal regularity-based motion score, and returns a single structured
output dictionary. It intentionally does not implement expert-vs-learner
comparison, DTW, or any persistence/API logic.
"""

from typing import Dict, List

from app.services.perception_service import run_perception_pipeline
from app.services.angle_metrics_service import compute_video_hand_angles
from app.services.trajectory_metrics_service import compute_video_trajectory_metrics
from app.services.velocity_metrics_service import compute_video_velocity_metrics
from app.services.scoring_service import compute_internal_motion_score


def validate_perception_output(perception_output: Dict) -> List[Dict]:
    """Validate perception output structure and return the frames list.

    Args:
        perception_output: Output dict from `run_perception_pipeline(video_path)`.

    Returns:
        The validated `frames` list.

    Raises:
        ValueError: if perception_output has an invalid structure.
    """
    if not isinstance(perception_output, dict):
        raise ValueError("validate_perception_output: perception_output must be a dict.")

    required_keys = {"video_path", "total_frames", "processed_frames", "frames"}
    missing = required_keys - set(perception_output.keys())
    if missing:
        missing_str = ", ".join(sorted(missing))
        raise ValueError(f"validate_perception_output: missing keys: {missing_str}.")

    frames = perception_output["frames"]
    if not isinstance(frames, list):
        raise ValueError("validate_perception_output: 'frames' must be a list.")

    return frames


def run_full_evaluation(video_path: str) -> Dict:
    """Run the full internal evaluation pipeline for a single video.

    Pipeline:
      1) perception: video -> frames -> hand landmarks
      2) angle metrics
      3) trajectory metrics
      4) velocity metrics
      5) internal-motion score

    Args:
        video_path: Path to the input video file.

    Returns:
        A dict with the exact structure specified in the STEP 6 requirements.
    """
    if not isinstance(video_path, str) or not video_path.strip():
        raise ValueError("run_full_evaluation: video_path must be a non-empty string.")

    perception_output = run_perception_pipeline(video_path)
    frames = validate_perception_output(perception_output)

    angle_frames = compute_video_hand_angles(frames)
    trajectory_metrics = compute_video_trajectory_metrics(frames)
    velocity_metrics = compute_video_velocity_metrics(frames)

    scoring_output = compute_internal_motion_score(angle_frames, trajectory_metrics, velocity_metrics)

    return {
        "video_path": video_path,
        "perception": perception_output,
        "angle_metrics": angle_frames,
        "trajectory_metrics": trajectory_metrics,
        "velocity_metrics": velocity_metrics,
        "scoring": scoring_output,
    }
