import asyncio

from app.services.normalization_service import normalize_coordinates


def test_normalize_coordinates_returns_input_unchanged() -> None:
    payload = {
        "video_id": "demo",
        "frames": [{"frame_index": 0, "positions": {}}],
    }

    result = asyncio.run(normalize_coordinates(payload))

    assert result == payload
