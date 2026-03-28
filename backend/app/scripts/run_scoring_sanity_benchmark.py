from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from typing import Any

from app.services.comparison_service import pair_motion_data_with_alignment
from app.services.evaluation_engine_service import build_metric_set
from app.services.scoring_service import compute_final_score
from app.services.temporal_alignment_service import align_sequences


@dataclass
class BenchmarkCase:
    name: str
    category: str
    expert_path: str
    learner_path: str
    expert_series: list[float]
    learner_series: list[float]
    low_signal: bool = False


def _build_modern_motion_payload(video_id: str, scalar_series: list[float], *, low_signal: bool = False) -> dict[str, Any]:
    frames: list[dict[str, Any]] = []
    for index, scalar in enumerate(scalar_series):
        present = not low_signal or (index % 4 == 0)
        magnitude = float(scalar if present else 0.0)
        hand_features = {
            "present": present,
            "joint_angles": {
                "wrist_index_mcp_index_tip": float(40.0 + magnitude),
                "wrist_middle_mcp_middle_tip": float(50.0 + magnitude),
                "wrist_ring_mcp_ring_tip": float(55.0 + magnitude),
                "wrist_index_mcp_pinky_mcp": float(60.0 + magnitude),
            },
            "wrist_position": [float(magnitude), float(magnitude * 0.5), 0.0],
            "palm_center": [float(magnitude * 0.8), float(magnitude * 0.4), 0.0],
            "fingertip_positions": {"index_tip": [float(magnitude * 1.2), float(magnitude * 0.6), 0.0]},
            "wrist_velocity": [float(magnitude * 0.2), float(magnitude * 0.1), 0.0],
            "palm_velocity": [float(magnitude * 0.15), float(magnitude * 0.08), 0.0],
            "hand_openness": float(0.5 + magnitude * 0.01),
            "pinch_distance": float(0.2 + magnitude * 0.01),
            "finger_spread": float(0.3 + magnitude * 0.01),
        }
        frames.append(
            {
                "frame_index": index,
                "timestamp_sec": index / 30.0,
                "flattened_feature_vector": [float(magnitude)] * 108,
                "left_hand_features": hand_features,
                "right_hand_features": hand_features,
            }
        )

    return {
        "video_id": video_id,
        "fps": 30.0,
        "total_frames": len(frames),
        "sequence_length": len(frames),
        "frames": frames,
    }


def _valid_landmark_frames(motion_payload: dict[str, Any]) -> int:
    valid = 0
    for frame in motion_payload.get("frames", []):
        left = frame.get("left_hand_features", {})
        right = frame.get("right_hand_features", {})
        if bool(left.get("present")) or bool(right.get("present")):
            valid += 1
    return valid


def _series_signal_std(values: list[float]) -> float:
    if not values:
        return 0.0
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / len(values)
    return math.sqrt(max(variance, 0.0))


def _build_cases() -> list[BenchmarkCase]:
    base = [0.0, 0.8, 1.5, 2.1, 2.8, 2.3, 1.7, 1.2, 0.7, 0.2]
    similar = [0.0, 0.5, 1.1, 1.8, 2.6, 2.0, 1.4, 1.0, 0.6, 0.2]
    unrelated = [2.8, 2.3, 2.0, 1.8, 1.5, 1.2, 0.9, 0.6, 0.3, 0.0]
    low_signal = [0.0] * 10

    return [
        BenchmarkCase(
            name="identical_vs_same_expert",
            category="identical video vs same expert",
            expert_path="storage/expert/benchmark/expert_base.mp4",
            learner_path="storage/expert/benchmark/expert_base.mp4",
            expert_series=base,
            learner_series=base,
        ),
        BenchmarkCase(
            name="same_task_different_performance",
            category="same task but different performance",
            expert_path="storage/expert/benchmark/expert_base.mp4",
            learner_path="storage/learner/benchmark/learner_similar.mp4",
            expert_series=base,
            learner_series=similar,
        ),
        BenchmarkCase(
            name="clearly_different_unrelated",
            category="clearly different / unrelated video",
            expert_path="storage/expert/benchmark/expert_base.mp4",
            learner_path="storage/learner/benchmark/learner_unrelated.mp4",
            expert_series=base,
            learner_series=unrelated,
        ),
        BenchmarkCase(
            name="invalid_low_signal",
            category="invalid / low-signal input",
            expert_path="storage/expert/benchmark/expert_base.mp4",
            learner_path="storage/learner/benchmark/learner_low_signal.mp4",
            expert_series=base,
            learner_series=low_signal,
            low_signal=True,
        ),
    ]


def _run_case(case: BenchmarkCase) -> dict[str, Any]:
    expert_motion = _build_modern_motion_payload(
        video_id=f"{case.name}_expert",
        scalar_series=case.expert_series,
    )
    learner_motion = _build_modern_motion_payload(
        video_id=f"{case.name}_learner",
        scalar_series=case.learner_series,
        low_signal=case.low_signal,
    )

    alignment = align_sequences(expert_motion=expert_motion, learner_motion=learner_motion)
    paired = pair_motion_data_with_alignment(
        expert_data=expert_motion,
        learner_data=learner_motion,
        aligned_pairs=alignment["aligned_pairs"],
    )
    dtw_norm_cost = float(alignment["dtw_normalized_cost"])
    metrics = build_metric_set(paired, dtw_normalized_cost=dtw_norm_cost).model_dump()
    score_payload = compute_final_score(metrics)

    expert_valid = _valid_landmark_frames(expert_motion)
    learner_valid = _valid_landmark_frames(learner_motion)
    expert_len = len(expert_motion["frames"])
    learner_len = len(learner_motion["frames"])
    min_valid_ratio = min(
        expert_valid / max(expert_len, 1),
        learner_valid / max(learner_len, 1),
    )

    confidence_flags = {
        "low_signal_detected": case.low_signal or _series_signal_std(case.learner_series) < 0.05,
        "min_valid_ratio": round(min_valid_ratio, 4),
        "sufficient_valid_frames": expert_valid >= 4 and learner_valid >= 4,
    }

    return {
        "name": case.name,
        "category": case.category,
        "expert_path": case.expert_path,
        "learner_path": case.learner_path,
        "valid_landmark_frames": {
            "expert": expert_valid,
            "learner": learner_valid,
        },
        "final_sequence_lengths": {
            "expert": expert_len,
            "learner": learner_len,
            "aligned_path_length": alignment["path_length"],
        },
        "dtw": {
            "raw_distance": round(float(alignment["dtw_total_cost"]), 6),
            "normalized_distance": round(float(alignment["dtw_normalized_cost"]), 6),
        },
        "metrics": {
            "angle_metric": metrics.get("angle_deviation"),
            "trajectory_metric": metrics.get("trajectory_deviation"),
            "velocity_metric": metrics.get("velocity_difference"),
            "smoothness_metric": metrics.get("smoothness_score"),
            "timing_metric": metrics.get("timing_score"),
            "hand_openness_metric": metrics.get("hand_openness_deviation"),
            "dtw_similarity": metrics.get("dtw_similarity"),
        },
        "final_score": score_payload.get("score"),
        "confidence_flags": confidence_flags,
    }


def _run_real_video_case(expert_path: str, learner_path: str, label: str) -> dict[str, Any]:
    """Run real MediaPipe -> motion -> DTW -> scoring on actual video files."""
    from app.services.motion_representation_service import process_video_to_motion_representation

    print(f"  Processing expert: {expert_path} ...")
    expert_motion = process_video_to_motion_representation(expert_path)
    expert_dict = expert_motion.model_dump()

    print(f"  Processing learner: {learner_path} ...")
    learner_motion = process_video_to_motion_representation(learner_path)
    learner_dict = learner_motion.model_dump()

    expert_valid = sum(1 for f in expert_motion.frames if f.left_hand_features.present or f.right_hand_features.present)
    learner_valid = sum(1 for f in learner_motion.frames if f.left_hand_features.present or f.right_hand_features.present)

    alignment = align_sequences(expert_motion=expert_dict, learner_motion=learner_dict)
    dtw_norm_cost = float(alignment["dtw_normalized_cost"])
    paired = pair_motion_data_with_alignment(
        expert_data=expert_dict,
        learner_data=learner_dict,
        aligned_pairs=alignment["aligned_pairs"],
    )
    metrics = build_metric_set(paired, dtw_normalized_cost=dtw_norm_cost).model_dump()
    score_payload = compute_final_score(metrics)

    return {
        "name": label,
        "category": "real_video",
        "expert_path": expert_path,
        "learner_path": learner_path,
        "valid_landmark_frames": {"expert": expert_valid, "learner": learner_valid},
        "final_sequence_lengths": {
            "expert": len(expert_dict["frames"]),
            "learner": len(learner_dict["frames"]),
            "aligned_path_length": alignment["path_length"],
        },
        "dtw": {
            "raw_distance": round(float(alignment["dtw_total_cost"]), 6),
            "normalized_distance": round(float(alignment["dtw_normalized_cost"]), 6),
        },
        "metrics": {
            "angle_metric": metrics.get("angle_deviation"),
            "trajectory_metric": metrics.get("trajectory_deviation"),
            "velocity_metric": metrics.get("velocity_difference"),
            "smoothness_metric": metrics.get("smoothness_score"),
            "timing_metric": metrics.get("timing_score"),
            "hand_openness_metric": metrics.get("hand_openness_deviation"),
            "dtw_similarity": metrics.get("dtw_similarity"),
        },
        "final_score": score_payload.get("score"),
        "confidence_flags": {
            "low_signal_detected": expert_valid < 3 or learner_valid < 3,
            "min_valid_ratio": round(
                min(
                    expert_valid / max(len(expert_dict["frames"]), 1),
                    learner_valid / max(len(learner_dict["frames"]), 1),
                ),
                4,
            ),
            "sufficient_valid_frames": expert_valid >= 3 and learner_valid >= 3,
        },
    }


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Scoring sanity benchmark")
    parser.add_argument("--real", action="store_true", help="Run with real video files via MediaPipe")
    parser.add_argument("--expert", type=str, default=None, help="Path to expert video (for --real mode)")
    parser.add_argument("--learner", type=str, default=None, help="Path to learner video (for --real mode)")
    args = parser.parse_args()

    if args.real:
        from pathlib import Path
        from app.core.config import settings

        expert_path = args.expert or str(settings.STORAGE_ROOT / "expert" / "pottery.mp4")
        learner_path = args.learner or expert_path

        if not Path(expert_path).exists():
            print(f"Expert video not found: {expert_path}")
            return
        if not Path(learner_path).exists():
            print(f"Learner video not found: {learner_path}")
            return

        print("=== Real Video Scoring Benchmark ===\n")

        same_label = "identical" if expert_path == learner_path else "expert_vs_learner"
        result = _run_real_video_case(expert_path, learner_path, same_label)
        print(json.dumps(result, indent=2))
        print("-" * 72)

        if expert_path == learner_path:
            print("(To test different videos, pass --learner <path>)")

        return

    print("=== Scoring Sanity Benchmark (synthetic) ===")
    print("Deterministic benchmark over controlled expert/learner pairs.\n")

    results = [_run_case(case) for case in _build_cases()]
    for result in results:
        print(json.dumps(result, indent=2))
        print("-" * 72)

    summary = {
        "case_count": len(results),
        "scores": {result["name"]: result["final_score"] for result in results},
    }
    print("Summary:")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
