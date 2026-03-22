import asyncio

from app.services.alignment_prep_service import prepare_alignment


def test_prepare_alignment_returns_stub_metadata() -> None:
    result = asyncio.run(prepare_alignment(expert_motion={}, learner_motion={}))

    assert result == {
        "aligned_pairs": [],
        "alignment_method": "none",
        "expert_length": 0,
        "learner_length": 0,
    }
