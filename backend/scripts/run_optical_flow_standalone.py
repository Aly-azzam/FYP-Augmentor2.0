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

    raw_result, summary_result = run_optical_flow_comparison(
        expert_video_path=expert_video_path,
        learner_video_path=learner_video_path,
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
        )

    similarities = evaluation["similarities"]
    optical_flow_score = evaluation["score"]["optical_flow_score"]

    es = summary_result.expert_summary
    ls = summary_result.learner_summary

    print(f"run_id:               {raw_result.run.run_id}")
    print(f"summary_json_path:    {summary_json_path}")
    print(f"raw_json_path:        {raw_json_path}")
    print(f"optical_flow_score:   {optical_flow_score}")
    print(f"magnitude_similarity: {similarities['magnitude_similarity']}")
    print(f"motion_area_similarity: {similarities['motion_area_similarity']}")
    print(f"angle_similarity:     {similarities['angle_similarity']}")

    print("\nexpert_summary:")
    print(f"  vibration_score:     {es.vibration_score}")
    print(f"  magnitude_jitter:    {es.magnitude_jitter}")
    print(f"  magnitude_std:       {es.magnitude_std}")

    print("\nlearner_summary:")
    print(f"  vibration_score:     {ls.vibration_score}")
    print(f"  magnitude_jitter:    {ls.magnitude_jitter}")
    print(f"  magnitude_std:       {ls.magnitude_std}")


if __name__ == "__main__":
    main()
