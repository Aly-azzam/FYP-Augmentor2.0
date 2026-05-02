from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from app.services.optical_flow.yolo_scissors_roi import (
    YoloScissorsROIConfig,
    YoloScissorsROIProvider,
    expand_scissors_bbox_to_roi,
    get_yolo_scissors_roi_for_frame,
)


@dataclass(slots=True)
class FakeDetection:
    bbox: list[float]
    confidence: float
    class_name: str | None = "scissors"


def test_expand_scissors_bbox_to_roi_clamps_to_frame() -> None:
    roi = expand_scissors_bbox_to_roi(
        [10.0, 20.0, 30.0, 60.0],
        frame_width=100,
        frame_height=80,
    )

    assert roi == [0, 4, 42, 80]


def test_get_yolo_scissors_roi_for_frame_returns_debug_fields() -> None:
    frame = np.zeros((80, 100, 3), dtype=np.uint8)

    result = get_yolo_scissors_roi_for_frame(
        frame=frame,
        frame_index=7,
        video_path="demo.mp4",
        detection_fn=lambda **_kwargs: FakeDetection(
            bbox=[10.0, 20.0, 30.0, 60.0],
            confidence=0.91,
        ),
    )

    assert result.roi_found is True
    assert result.roi_source == "yolo_scissors_expanded"
    assert result.original_scissors_bbox == [10.0, 20.0, 30.0, 60.0]
    assert result.expanded_roi_bbox == [0, 4, 42, 80]
    assert result.expanded_roi_bbox_raw == [0, 4, 42, 80]
    assert result.expanded_roi_bbox_smoothed == [0, 4, 42, 80]
    assert result.detection_confidence == 0.91
    assert result.roi_reused_from_previous is False
    assert result.fallback_reason is None
    assert result.to_debug_dict()["video_path"] == "demo.mp4"


def test_provider_reuses_previous_roi_when_yolo_misses() -> None:
    frame = np.zeros((80, 100, 3), dtype=np.uint8)
    calls = {"count": 0}

    def detector(**_kwargs):
        calls["count"] += 1
        if calls["count"] == 1:
            return FakeDetection(
                bbox=[10.0, 20.0, 30.0, 60.0],
                confidence=0.91,
            )
        raise RuntimeError("no scissors")

    provider = YoloScissorsROIProvider(
        video_path="demo.mp4",
        config=YoloScissorsROIConfig(max_roi_hold_frames=5),
        detection_fn=detector,
    )

    first_roi = provider.detect(frame, frame_index=1)
    second_roi = provider.detect(frame, frame_index=2)
    debug = provider.last_debug_dict()

    assert first_roi == (0, 4, 42, 80)
    assert second_roi == first_roi
    assert debug is not None
    assert debug["roi_found"] is True
    assert debug["roi_reused_from_previous"] is True
    assert debug["roi_source"] == "previous_yolo_bbox"
    assert debug["expanded_roi_bbox_smoothed"] == [0, 4, 42, 80]
    assert debug["fallback_reason"].startswith("yolo_missing:")


def test_get_yolo_scissors_roi_for_frame_falls_back_when_hold_expired() -> None:
    frame = np.zeros((80, 100, 3), dtype=np.uint8)

    result = get_yolo_scissors_roi_for_frame(
        frame=frame,
        frame_index=10,
        video_path="demo.mp4",
        previous_bbox=[0, 4, 42, 80],
        frames_since_previous_bbox=5,
        config=YoloScissorsROIConfig(max_roi_hold_frames=5),
        detection_fn=lambda **_kwargs: (_ for _ in ()).throw(RuntimeError("no scissors")),
    )

    assert result.roi_found is False
    assert result.expanded_roi_bbox is None
    assert result.expanded_roi_bbox_smoothed is None
    assert result.roi_reused_from_previous is False
    assert result.fallback_reason is not None
    assert result.fallback_reason.startswith("roi_hold_expired:")


def test_provider_smooths_expanded_roi_between_yolo_detections() -> None:
    frame = np.zeros((80, 100, 3), dtype=np.uint8)
    detections = [
        FakeDetection(bbox=[10.0, 20.0, 30.0, 60.0], confidence=0.91),
        FakeDetection(bbox=[20.0, 20.0, 40.0, 60.0], confidence=0.92),
    ]

    def detector(**_kwargs):
        return detections.pop(0)

    provider = YoloScissorsROIProvider(
        video_path="demo.mp4",
        config=YoloScissorsROIConfig(
            roi_smoothing_enabled=True,
            roi_smoothing_alpha=0.5,
        ),
        detection_fn=detector,
    )

    first_roi = provider.detect(frame, frame_index=1)
    second_roi = provider.detect(frame, frame_index=2)
    debug = provider.last_debug_dict()

    assert first_roi == (0, 4, 42, 80)
    assert second_roi == (4, 4, 47, 80)
    assert debug is not None
    assert debug["expanded_roi_bbox_raw"] == [8, 4, 52, 80]
    assert debug["expanded_roi_bbox_smoothed"] == [4, 4, 47, 80]

