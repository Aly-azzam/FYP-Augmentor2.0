import asyncio

from app.services.motion_features_service import compute_kinematics


def test_compute_kinematics_returns_empty_stub_payload() -> None:
    result = asyncio.run(compute_kinematics(sequence_data={}, fps=30.0))

    assert result == {
        "velocity": {},
        "speed": {},
    }
