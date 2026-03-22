"""MediaPipe Hands Service (Perception Engine).

This module encapsulates MediaPipe Hands usage for:
- initializing the detector
- processing RGB frames
- extracting left/right hand landmarks with readable landmark names
- returning clean structured data for later formatting

It intentionally does not implement any API logic or the full perception pipeline.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional

import mediapipe as mp


_LANDMARK_NAMES: List[str] = [
    "wrist",
    "thumb_cmc",
    "thumb_mcp",
    "thumb_ip",
    "thumb_tip",
    "index_finger_mcp",
    "index_finger_pip",
    "index_finger_dip",
    "index_finger_tip",
    "middle_finger_mcp",
    "middle_finger_pip",
    "middle_finger_dip",
    "middle_finger_tip",
    "ring_finger_mcp",
    "ring_finger_pip",
    "ring_finger_dip",
    "ring_finger_tip",
    "pinky_mcp",
    "pinky_pip",
    "pinky_dip",
    "pinky_tip",
]


def initialize_hands_detector(
    static_image_mode: bool = False,
    max_num_hands: int = 2,
    min_detection_confidence: float = 0.5,
    min_tracking_confidence: float = 0.5,
):
    """Initialize and return a MediaPipe Hands detector."""
    return mp.solutions.hands.Hands(
        static_image_mode=static_image_mode,
        max_num_hands=max_num_hands,
        min_detection_confidence=min_detection_confidence,
        min_tracking_confidence=min_tracking_confidence,
    )


def process_frame(frame_rgb: Any, hands_detector: Any) -> Any:
    """Process a single RGB frame and return raw MediaPipe results."""
    if frame_rgb is None:
        raise ValueError("process_frame: frame_rgb must not be None.")
    if hands_detector is None:
        raise ValueError("process_frame: hands_detector must not be None.")
    return hands_detector.process(frame_rgb)


def extract_hand_landmarks(results: Any) -> dict:
    """Extract left/right hand landmarks from MediaPipe results.

    Returns:
        A dict shaped exactly like:
        {
          "left_hand": dict | None,
          "right_hand": dict | None
        }
    """
    if results is None:
        raise ValueError("extract_hand_landmarks: results must not be None.")

    left_hand: Optional[Dict[str, List[float]]] = None
    right_hand: Optional[Dict[str, List[float]]] = None

    multi_landmarks = getattr(results, "multi_hand_landmarks", None)
    if not multi_landmarks:
        return {"left_hand": None, "right_hand": None}

    multi_handedness = getattr(results, "multi_handedness", None) or []
    num_hands = len(multi_landmarks)

    for i in range(num_hands):
        # Determine left/right from handedness classification.
        hand_label: Optional[str] = None
        if i < len(multi_handedness):
            handedness = multi_handedness[i]
            classifications = getattr(handedness, "classification", None) or []
            if classifications:
                hand_label = getattr(classifications[0], "label", None)

        if hand_label not in ("Left", "Right"):
            raise ValueError(
                "extract_hand_landmarks: missing/invalid handedness label "
                f"for hand index {i}: {hand_label!r}"
            )

        hand_landmarks_obj = multi_landmarks[i]
        landmark_list = getattr(hand_landmarks_obj, "landmark", None)
        if not landmark_list or len(landmark_list) != 21:
            raise ValueError(
                "extract_hand_landmarks: expected 21 landmarks per hand, "
                f"got {0 if not landmark_list else len(landmark_list)}"
            )

        hand_dict: Dict[str, List[float]] = {}
        for lm_name, lm in zip(_LANDMARK_NAMES, landmark_list):
            # Preserve MediaPipe normalized coordinates as raw floats (no rounding).
            x = float(lm.x)
            y = float(lm.y)
            z = float(lm.z)

            # Guard against NaNs to avoid propagating invalid values silently.
            if any(math.isnan(v) for v in (x, y, z)):
                raise ValueError(f"extract_hand_landmarks: landmark contains NaN for {lm_name}.")

            hand_dict[lm_name] = [x, y, z]

        if hand_label == "Left":
            left_hand = hand_dict
        else:
            right_hand = hand_dict

    return {"left_hand": left_hand, "right_hand": right_hand}


def process_video_frames(frames: list[dict], hands_detector: Any) -> list[dict]:
    """Process a list of preprocessed frame dicts into landmark dicts."""
    if hands_detector is None:
        raise ValueError("process_video_frames: hands_detector must not be None.")
    if frames is None:
        raise ValueError("process_video_frames: frames must not be None.")
    if not isinstance(frames, list):
        raise ValueError("process_video_frames: frames must be a list of dicts.")

    output_frames: list[dict] = []
    for frame in frames:
        if not isinstance(frame, dict):
            raise ValueError("process_video_frames: each frame must be a dict.")

        if "frame_index" not in frame or "timestamp" not in frame or "frame_rgb" not in frame:
            raise ValueError(
                "process_video_frames: frame dict must include keys "
                "'frame_index', 'timestamp', and 'frame_rgb'."
            )

        frame_index = frame["frame_index"]
        timestamp = frame["timestamp"]
        frame_rgb = frame["frame_rgb"]

        results = process_frame(frame_rgb=frame_rgb, hands_detector=hands_detector)
        hand_landmarks = extract_hand_landmarks(results)

        output_frames.append(
            {
                "frame_index": frame_index,
                "timestamp": timestamp,
                "left_hand": hand_landmarks["left_hand"],
                "right_hand": hand_landmarks["right_hand"],
            }
        )

    return output_frames
