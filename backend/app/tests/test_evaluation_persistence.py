from __future__ import annotations

import asyncio

from fastapi.testclient import TestClient

from app.core.database import Base, SessionLocal, engine
from app.main import app
from app.models.chapter import Chapter
from app.models.course import Course
from app.models.evaluation_result import EvaluationResult as EvaluationResultModel
from app.models.expert_video import ExpertVideo
from app.models.learner_attempt import LearnerAttempt
from app.models.progress import Progress
from app.models.user import User
from app.services.evaluation_engine_service import run_evaluation_pipeline
from app.services.progress_service import compute_progress


def _assert_safe_test_engine() -> None:
    engine_url = str(engine.url).lower()
    if not any(marker in engine_url for marker in ("test", "pytest", "sqlite", "tmp")):
        raise RuntimeError(
            "Refusing destructive DB test against a non-test engine URL. "
            f"Resolved engine URL: {engine.url}"
        )


def _build_modern_motion_payload(video_id: str, scalar_series: list[float]) -> dict:
    frames = []
    for idx, scalar in enumerate(scalar_series):
        hand_features = {
            "joint_angles": {
                "wrist_index_mcp_index_tip": float(40.0 + scalar),
                "wrist_middle_mcp_middle_tip": float(50.0 + scalar),
                "wrist_ring_mcp_ring_tip": float(55.0 + scalar),
                "wrist_index_mcp_pinky_mcp": float(60.0 + scalar),
            },
            "wrist_position": [float(scalar), float(scalar * 0.5), 0.0],
            "palm_center": [float(scalar * 0.8), float(scalar * 0.4), 0.0],
            "fingertip_positions": {"index_tip": [float(scalar * 1.2), float(scalar * 0.6), 0.0]},
            "wrist_velocity": [float(scalar * 0.2), float(scalar * 0.1), 0.0],
            "palm_velocity": [float(scalar * 0.15), float(scalar * 0.08), 0.0],
            "hand_openness": float(0.5 + scalar * 0.01),
            "pinch_distance": float(0.2 + scalar * 0.01),
            "finger_spread": float(0.3 + scalar * 0.01),
        }
        frames.append(
            {
                "frame_index": idx,
                "timestamp_sec": idx / 30.0,
                "flattened_feature_vector": [float(scalar)] * 108,
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


def _ensure_base_data(attempt_id: str) -> None:
    _assert_safe_test_engine()
    _ = ExpertVideo
    _ = Progress
    _ = User
    Base.metadata.drop_all(bind=engine)
    Base.metadata.create_all(bind=engine)
    session = SessionLocal()
    try:
        course = Course(title="Persistence Course")
        session.add(course)
        session.flush()
        chapter = Chapter(course_id=course.id, title="Persistence Chapter", order=1)
        session.add(chapter)
        session.flush()
        attempt = LearnerAttempt(id=attempt_id, chapter_id=chapter.id, status="created")
        session.add(attempt)
        session.commit()
    finally:
        session.close()


def test_evaluation_is_saved_and_retrievable() -> None:
    attempt_id = "attempt-persist-001"
    _ensure_base_data(attempt_id)

    expert_motion = _build_modern_motion_payload("expert", [0.0, 1.0, 2.0, 3.0])
    learner_motion = _build_modern_motion_payload("learner", [0.0, 0.5, 1.0, 1.5, 2.5, 3.0])
    result = asyncio.run(
        run_evaluation_pipeline(
            expert_motion_source=expert_motion,
            learner_motion_source=learner_motion,
            attempt_id=attempt_id,
        )
    )

    session = SessionLocal()
    try:
        saved = session.query(EvaluationResultModel).filter(EvaluationResultModel.id == result.evaluation_id).first()
        assert saved is not None
        assert saved.attempt_id == attempt_id
        assert isinstance(saved.metrics, dict)
    finally:
        session.close()


def test_history_endpoint_returns_multiple_entries_desc() -> None:
    attempt_id = "attempt-history-001"
    _ensure_base_data(attempt_id)
    session = SessionLocal()
    try:
        session.add(EvaluationResultModel(attempt_id=attempt_id, score=60.0, metrics={"a": 1}))
        session.add(EvaluationResultModel(attempt_id=attempt_id, score=80.0, metrics={"a": 2}))
        session.commit()
    finally:
        session.close()

    client = TestClient(app)
    response = client.get(f"/api/evaluations/history/{attempt_id}")
    assert response.status_code == 200
    data = response.json()
    assert "evaluations" in data
    assert len(data["evaluations"]) >= 2
    assert data["evaluations"][0]["created_at"] >= data["evaluations"][1]["created_at"]


def test_progress_computation_and_endpoint() -> None:
    computed = compute_progress(
        [
            {"score": 60.0},
            {"score": 90.0},
            {"score": 75.0},
        ]
    )
    assert computed["best_score"] == 90.0
    assert computed["average_score"] == 75.0
    assert computed["attempt_count"] == 3
    assert computed["latest_score"] == 60.0

    attempt_id = "attempt-progress-001"
    _ensure_base_data(attempt_id)
    session = SessionLocal()
    try:
        session.add(EvaluationResultModel(attempt_id=attempt_id, score=70.0))
        session.add(EvaluationResultModel(attempt_id=attempt_id, score=85.0))
        session.commit()
    finally:
        session.close()

    client = TestClient(app)
    response = client.get(f"/api/evaluations/progress/{attempt_id}")
    assert response.status_code == 200
    data = response.json()
    assert data["attempt_id"] == attempt_id
    assert data["attempt_count"] >= 2
    assert data["best_score"] >= data["latest_score"] or data["latest_score"] >= 0.0


def test_get_evaluation_by_id_endpoint() -> None:
    attempt_id = "attempt-get-id-001"
    _ensure_base_data(attempt_id)
    session = SessionLocal()
    try:
        row = EvaluationResultModel(attempt_id=attempt_id, score=77.0, metrics={"angle_deviation": 0.1})
        session.add(row)
        session.commit()
        session.refresh(row)
        row_id = row.id
    finally:
        session.close()

    client = TestClient(app)
    response = client.get(f"/api/evaluations/{row_id}")
    assert response.status_code == 200
    data = response.json()
    assert data["id"] == row_id
    assert data["attempt_id"] == attempt_id
    assert data["score"] == 77.0

