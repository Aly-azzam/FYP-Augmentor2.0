from __future__ import annotations

import asyncio
import json
from pathlib import Path

from app.services.evaluation_engine_service import run_evaluation_pipeline


BACKEND_DIR = Path(__file__).resolve().parents[2]
MOTION_DIR = BACKEND_DIR / "storage" / "processed" / "motion"
EXPERT_MOCK_PATH = MOTION_DIR / "expert_mock.json"
LEARNER_MOCK_PATH = MOTION_DIR / "learner_mock.json"


async def main() -> None:
    try:
        result = await run_evaluation_pipeline(
            expert_motion_source=EXPERT_MOCK_PATH,
            learner_motion_source=LEARNER_MOCK_PATH,
        )

        print("\n=== Evaluation Demo Result ===")
        print(f"evaluation_id: {result.evaluation_id}")
        print(f"score: {result.score}")

        print("\nmetrics:")
        print(f"  angle_deviation: {result.metrics.angle_deviation}")
        print(f"  trajectory_deviation: {result.metrics.trajectory_deviation}")
        print(f"  velocity_difference: {result.metrics.velocity_difference}")
        print(f"  tool_alignment_deviation: {result.metrics.tool_alignment_deviation}")

        print("\nsummary:")
        print(f"  main_strength: {result.summary.main_strength}")
        print(f"  main_weakness: {result.summary.main_weakness}")
        print(f"  focus_area: {result.summary.focus_area}")

        print("\nvlm_payload:")
        print(json.dumps(result.vlm_payload.model_dump(), indent=2))

    except Exception as exc:
        print(f"Evaluation demo failed: {exc}")


if __name__ == "__main__":
    asyncio.run(main())
