from __future__ import annotations

import asyncio
import json
import uuid
from pathlib import Path

from app.core.database import SessionLocal
from app.models.chapter import Chapter
from app.models.course import Course
from app.models.evaluation_result import EvaluationResult as EvaluationResultModel
from app.models.expert_video import ExpertVideo
from app.models.learner_attempt import LearnerAttempt
from app.models.user import User
from app.services.evaluation_engine_service import run_evaluation_pipeline


BACKEND_DIR = Path(__file__).resolve().parents[2]
MOTION_DIR = BACKEND_DIR / "storage" / "processed" / "motion"
EXPERT_MOCK_PATH = MOTION_DIR / "expert_mock.json"
LEARNER_MOCK_PATH = MOTION_DIR / "learner_mock.json"


def _build_modern_motion_payload(video_id: str, scalar_series: list[float]) -> dict:
    frames = []
    for idx, scalar in enumerate(scalar_series):
        frame_vector = [float(scalar)] * 108
        hand_joint_angles = {
            "wrist_index_mcp_index_tip": float(40.0 + scalar),
            "wrist_middle_mcp_middle_tip": float(50.0 + scalar),
        }
        hand_features = {
            "joint_angles": hand_joint_angles,
            "wrist_position": [float(scalar), float(scalar * 0.5), 0.0],
            "palm_center": [float(scalar * 0.8), float(scalar * 0.4), 0.0],
            "fingertip_positions": {
                "index_tip": [float(scalar * 1.2), float(scalar * 0.6), 0.0],
            },
            "wrist_velocity": [float(scalar * 0.2), float(scalar * 0.1), 0.0],
            "palm_velocity": [float(scalar * 0.15), float(scalar * 0.08), 0.0],
        }
        frames.append(
            {
                "frame_index": idx,
                "timestamp_sec": idx / 30.0,
                "flattened_feature_vector": frame_vector,
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


async def main() -> None:
    try:
        # Use modern Step 2.4-style payloads so demo runs DTW alignment path.
        expert_motion = _build_modern_motion_payload(
            video_id="expert-modern-demo",
            scalar_series=[0.0, 1.0, 2.0, 3.0],
        )
        learner_motion = _build_modern_motion_payload(
            video_id="learner-modern-demo",
            scalar_series=[0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0],
        )
        attempt_id = _ensure_demo_attempt()
        print("Using modern aligned path: True")
        result = await run_evaluation_pipeline(
            expert_motion_source=expert_motion,
            learner_motion_source=learner_motion,
            attempt_id=attempt_id,
        )

        print("\n=== Evaluation Demo Result ===")
        print(f"evaluation_id: {result.evaluation_id}")
        print(f"score: {result.score}")
        print(f"saved_evaluation_id: {result.evaluation_id}")

        print("\nmetrics:")
        print(f"  angle_deviation: {result.metrics.angle_deviation}")
        print(f"  trajectory_deviation: {result.metrics.trajectory_deviation}")
        print(f"  velocity_difference: {result.metrics.velocity_difference}")
        print(f"  smoothness_score: {result.metrics.smoothness_score}")
        print(f"  timing_score: {result.metrics.timing_score}")
        print(f"  hand_openness_deviation: {result.metrics.hand_openness_deviation}")
        print(f"  tool_alignment_deviation: {result.metrics.tool_alignment_deviation}")

        print("\nper_metric_breakdown:")
        contribution_sum = 0.0
        for metric_name, item in result.per_metric_breakdown.items():
            contribution_sum += float(item.contribution)
            print(f"  {metric_name}:")
            print(f"    raw_value: {item.raw_value}")
            print(f"    normalized_quality: {item.normalized_quality}")
            print(f"    weight: {item.weight}")
            print(f"    contribution: {item.contribution}")
        print(f"\ncontribution_sum: {round(contribution_sum, 4)}")

        print("\nkey_error_moments:")
        for moment in result.key_error_moments:
            semantic_text = moment.semantic_label or moment.label
            print(
                f"  - {semantic_text}: "
                f"aligned {moment.start_aligned_index}-{moment.end_aligned_index}, "
                f"expert {moment.start_expert_frame}-{moment.end_expert_frame}, "
                f"learner {moment.start_learner_frame}-{moment.end_learner_frame}, "
                f"severity={moment.severity}"
            )

        print("\nSemantic phases:")
        print("  expert:")
        expert_semantics = result.semantic_phases.get("expert")
        if expert_semantics:
            for phase in expert_semantics.phases:
                print(f"    - {phase.label}: frames {phase.start_frame}-{phase.end_frame}")
        print("  learner:")
        learner_semantics = result.semantic_phases.get("learner")
        if learner_semantics:
            for phase in learner_semantics.phases:
                print(f"    - {phase.label}: frames {phase.start_frame}-{phase.end_frame}")

        print("\nsummary:")
        print(f"  main_strength: {result.summary.main_strength}")
        print(f"  main_weakness: {result.summary.main_weakness}")
        print(f"  focus_area: {result.summary.focus_area}")

        print("\nExplanation:")
        print(f"  mode: {result.explanation.mode}")
        print(f"  full: {result.explanation.explanation}")
        print(f"  strengths: {result.explanation.strengths}")
        print(f"  weaknesses: {result.explanation.weaknesses}")
        print(f"  advice: {result.explanation.advice}")

        print("\nvlm_payload:")
        print(json.dumps(result.vlm_payload.model_dump(), indent=2))

        session = SessionLocal()
        try:
            saved = (
                session.query(EvaluationResultModel)
                .filter(EvaluationResultModel.id == result.evaluation_id)
                .first()
            )
            print("\nSaved record fetched from DB:")
            if saved is None:
                print("  not found")
            else:
                print(
                    json.dumps(
                        {
                            "id": saved.id,
                            "attempt_id": saved.attempt_id,
                            "score": saved.score,
                            "metrics": saved.metrics,
                            "created_at": str(saved.created_at),
                        },
                        indent=2,
                    )
                )
        finally:
            session.close()

    except Exception as exc:
        print(f"Evaluation demo failed: {exc}")


def _ensure_demo_attempt() -> str:
    _ = ExpertVideo
    _ = User
    session = SessionLocal()
    try:
        existing = (
            session.query(LearnerAttempt)
            .filter(LearnerAttempt.original_filename == "demo-attempt")
            .first()
        )
        if existing:
            return str(existing.id)

        course = Course(title="Demo Course")
        session.add(course)
        session.flush()
        chapter = Chapter(course_id=course.id, title="Demo Chapter", order=1)
        session.add(chapter)
        session.flush()
        attempt = LearnerAttempt(
            id=str(uuid.uuid4()),
            chapter_id=chapter.id,
            status="created",
            original_filename="demo-attempt",
        )
        session.add(attempt)
        session.commit()
        return str(attempt.id)
    finally:
        session.close()


if __name__ == "__main__":
    asyncio.run(main())
