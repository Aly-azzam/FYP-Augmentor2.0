from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

# Optical Flow does not need the DB; provide a safe fallback for the env-var.
os.environ.setdefault("DATABASE_URL", "sqlite:///./augmentor.db")

from app.services.optical_flow import (  # noqa: E402
    FarnebackConfig,
    evaluate_optical_flow_summary,
    run_optical_flow_comparison,
)
from app.services.optical_flow.io_utils import save_optical_flow_results  # noqa: E402
from app.services.optical_flow.visualizer import (  # noqa: E402
    create_comparison_visualizations,
)


OUTPUT_DIR = BACKEND_ROOT / "storage" / "outputs" / "optical_flow" / "test_runs"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run standalone Optical Flow comparison on expert and learner videos.",
    )
    parser.add_argument(
        "--expert-video",
        required=True,
        help="Path to the expert/reference video.",
    )
    parser.add_argument(
        "--learner-video",
        required=True,
        help="Path to the learner/practice video.",
    )
    parser.add_argument(
        "--save-visualizations",
        action="store_true",
        help="Also save HSV optical flow visualization videos.",
    )
    parser.add_argument(
        "--use-hand-roi",
        action="store_true",
        help="Run Farneback only inside detected MediaPipe hand ROIs when possible.",
    )
    parser.add_argument(
        "--roi-padding-px",
        type=int,
        default=40,
        help="Padding in pixels around the detected hand ROI.",
    )
    parser.add_argument(
        "--roi-enlarge-padding-px",
        type=int,
        default=50,
        help="Minimum effective padding in pixels for the locked hand ROI.",
    )
    parser.add_argument(
        "--roi-hand-preference",
        choices=("right", "left", "largest", "first"),
        default="right",
        help="Which detected hand to track for ROI cropping.",
    )
    parser.add_argument(
        "--roi-lock-target",
        nargs="?",
        const="right",
        default="right",
        choices=("right", "left", "largest", "first", "none"),
        help="Lock ROI onto this target hand. Use without a value to lock right.",
    )
    parser.add_argument(
        "--no-roi-lock-target",
        dest="roi_lock_target",
        action="store_const",
        const="none",
        help="Disable target locking.",
    )
    parser.add_argument(
        "--roi-lock-strict",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="When target lock is lost, avoid switching to another hand.",
    )
    parser.add_argument(
        "--roi-lock-max-missing-frames",
        type=int,
        default=5,
        help="Frames to reuse the locked ROI when the target is temporarily missing.",
    )
    parser.add_argument(
        "--roi-lock-max-center-distance-ratio",
        type=float,
        default=0.3,
        help="Maximum accepted locked-target center movement as a ratio of frame width.",
    )
    return parser


def _resolve_path(raw_path: str) -> Path:
    path = Path(raw_path)
    if path.is_absolute():
        return path
    return (BACKEND_ROOT / path).resolve()


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    expert_video_path = _resolve_path(args.expert_video)
    learner_video_path = _resolve_path(args.learner_video)
    farneback_config = FarnebackConfig(
        use_hand_roi=args.use_hand_roi,
        roi_padding_px=args.roi_padding_px,
        roi_enlarge_padding_px=args.roi_enlarge_padding_px,
        roi_hand_preference=args.roi_hand_preference,
        roi_lock_target=args.roi_lock_target,
        roi_lock_strict=args.roi_lock_strict,
        roi_lock_max_missing_frames=args.roi_lock_max_missing_frames,
        roi_lock_max_center_distance_ratio=args.roi_lock_max_center_distance_ratio,
    )

    raw_result, summary_result = run_optical_flow_comparison(
        expert_video_path=expert_video_path,
        learner_video_path=learner_video_path,
        config=farneback_config,
    )

    evaluation = evaluate_optical_flow_summary(summary_result)
    raw_json_path, summary_json_path = save_optical_flow_results(
        raw_result=raw_result,
        summary_result=summary_result,
        output_dir=OUTPUT_DIR,
    )

    if args.save_visualizations:
        create_comparison_visualizations(
            expert_video_path=expert_video_path,
            learner_video_path=learner_video_path,
            output_dir=OUTPUT_DIR / "visualizations",
            run_id=raw_result.run.run_id,
            config=farneback_config,
        )

    similarities = evaluation["similarities"]
    optical_flow_score = evaluation["score"]["optical_flow_score"]

    es = summary_result.expert_summary
    ls = summary_result.learner_summary
    metrics = summary_result.comparison_metrics

    print(f"run_id:               {raw_result.run.run_id}")
    print(f"summary_json_path:    {summary_json_path}")
    print(f"raw_json_path:        {raw_json_path}")
    print(f"optical_flow_score:   {optical_flow_score}")
    print(f"magnitude_similarity: {similarities['magnitude_similarity']}")
    print(f"motion_area_similarity: {similarities['motion_area_similarity']}")
    print(f"angle_similarity:     {similarities['angle_similarity']}")
    print(f"vibration_difference: {metrics.vibration_difference}")
    print(f"vibration_ratio:      {metrics.vibration_ratio}")

    print("\nexpert_summary:")
    print(f"  vibration_score:     {es.vibration_score}")
    print(f"  magnitude_jitter:    {es.magnitude_jitter}")
    print(f"  magnitude_std:       {es.magnitude_std}")
    print(f"  vibration_high_freq_mean: {es.vibration_high_freq_mean}")
    print(f"  vibration_high_freq_max:  {es.vibration_high_freq_max}")
    print(f"  roi_usage_ratio:     {es.roi_usage_ratio}")
    print(f"  roi_frames_used:     {es.roi_frames_used}")
    print(f"  roi_fallback_frames: {es.roi_fallback_frames}")

    print("\nlearner_summary:")
    print(f"  vibration_score:     {ls.vibration_score}")
    print(f"  magnitude_jitter:    {ls.magnitude_jitter}")
    print(f"  magnitude_std:       {ls.magnitude_std}")
    print(f"  vibration_high_freq_mean: {ls.vibration_high_freq_mean}")
    print(f"  vibration_high_freq_max:  {ls.vibration_high_freq_max}")
    print(f"  roi_usage_ratio:     {ls.roi_usage_ratio}")
    print(f"  roi_frames_used:     {ls.roi_frames_used}")
    print(f"  roi_fallback_frames: {ls.roi_fallback_frames}")


if __name__ == "__main__":
    main()
