import pytest


dtw_utils = pytest.importorskip("app.utils.dtw_utils")


def test_compute_dtw_returns_expected_shape() -> None:
    result = dtw_utils.compute_dtw(
        [[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]],
        [[0.0, 0.0], [2.0, 2.0]],
    )

    assert "distance" in result
    assert "normalized_distance" in result
    assert "path" in result
    assert isinstance(result["path"], list)
