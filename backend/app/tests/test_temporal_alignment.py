from __future__ import annotations

from app.services.temporal_alignment_service import align_sequences


def _frame_vector(value: float, dim: int = 108) -> list[float]:
    return [float(value)] * dim


def _build_motion_payload(values: list[float]) -> dict:
    return {
        "video_id": "demo-video",
        "fps": 30.0,
        "total_frames": len(values),
        "sequence_length": len(values),
        "frames": [
            {
                "frame_index": idx,
                "timestamp_sec": idx / 30.0,
                "flattened_feature_vector": _frame_vector(val),
            }
            for idx, val in enumerate(values)
        ],
    }


def test_identical_sequences_have_near_zero_cost() -> None:
    expert_motion = _build_motion_payload([0.0, 1.0, 2.0, 3.0])
    learner_motion = _build_motion_payload([0.0, 1.0, 2.0, 3.0])

    result = align_sequences(expert_motion, learner_motion)

    assert result["dtw_total_cost"] < 1e-9
    assert result["dtw_normalized_cost"] < 1e-9
    assert result["alignment_path"][0] == [0, 0]
    assert result["alignment_path"][-1] == [3, 3]


def test_same_motion_different_speed_produces_valid_alignment() -> None:
    expert_motion = _build_motion_payload([0.0, 1.0, 2.0, 3.0])
    learner_motion = _build_motion_payload([0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0])

    result = align_sequences(expert_motion, learner_motion)

    assert result["path_length"] >= max(
        len(expert_motion["frames"]),
        len(learner_motion["frames"]),
    )
    assert result["dtw_total_cost"] >= 0.0
    assert result["dtw_normalized_cost"] >= 0.0
    assert result["alignment_path"][0] == [0, 0]
    assert result["alignment_path"][-1] == [3, 6]


def test_output_structure_is_correct() -> None:
    expert_motion = _build_motion_payload([0.0, 1.0])
    learner_motion = _build_motion_payload([0.0, 1.0, 2.0])

    result = align_sequences(expert_motion, learner_motion)

    assert set(result.keys()) == {
        "alignment_path",
        "aligned_pairs",
        "dtw_total_cost",
        "dtw_normalized_cost",
        "path_length",
    }
    assert isinstance(result["alignment_path"], list)
    assert isinstance(result["aligned_pairs"], list)
    assert len(result["alignment_path"]) == result["path_length"]
    assert len(result["aligned_pairs"]) == result["path_length"]
    assert isinstance(result["dtw_total_cost"], float)
    assert isinstance(result["dtw_normalized_cost"], float)


def test_alignment_path_is_monotonic() -> None:
    expert_motion = _build_motion_payload([0.0, 1.0, 2.0, 3.0, 4.0])
    learner_motion = _build_motion_payload([0.0, 0.0, 1.0, 1.0, 2.0, 3.0, 4.0])

    result = align_sequences(expert_motion, learner_motion)
    path = result["alignment_path"]

    for previous, current in zip(path, path[1:]):
        assert current[0] >= previous[0]
        assert current[1] >= previous[1]
        assert (current[0] - previous[0]) in (0, 1)
        assert (current[1] - previous[1]) in (0, 1)
