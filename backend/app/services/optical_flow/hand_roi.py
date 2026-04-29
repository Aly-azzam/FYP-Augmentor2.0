from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional, Tuple

import cv2
import numpy as np

try:
    import mediapipe as mp  # type: ignore

    _MEDIAPIPE_AVAILABLE = True
except ImportError:
    mp = None  # type: ignore[assignment]
    _MEDIAPIPE_AVAILABLE = False


# (x1, y1, x2, y2), using NumPy's inclusive-start / exclusive-end convention.
ROIBox = Tuple[int, int, int, int]
ROICenter = Tuple[float, float]
ROIHandPreference = Literal["right", "left", "largest", "first"]
ROILockTarget = Literal["right", "left", "largest", "first"]


@dataclass
class _DetectedHandROI:
    roi: ROIBox
    label: str | None

    @property
    def area(self) -> int:
        x1, y1, x2, y2 = self.roi
        return max(0, x2 - x1) * max(0, y2 - y1)

    @property
    def center(self) -> ROICenter:
        x1, y1, x2, y2 = self.roi
        return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


class HandROIDetector:
    """
    Detect hand bounding boxes per frame for optional optical-flow ROI cropping.

    The current AugMentor backend uses MediaPipe Tasks, so that path is
    preferred. The legacy ``mp.solutions.hands`` path is kept as a compatibility
    fallback for environments that expose it.
    """

    def __init__(
        self,
        padding_px: int = 40,
        hand_preference: ROIHandPreference = "right",
        max_missing_frames: int = 5,
        lock_target: bool | str = "right",
        lock_max_missing_frames: int = 5,
        lock_max_center_distance_ratio: float = 0.3,
        lock_strict: bool = True,
    ) -> None:
        if not _MEDIAPIPE_AVAILABLE:
            raise RuntimeError(
                "mediapipe is not installed. Install it before enabling hand ROI."
            )

        self._padding_px = max(0, int(padding_px))
        self._hand_preference = self._normalize_hand_preference(hand_preference)
        self._max_missing_frames = max(0, int(max_missing_frames))
        self._previous_roi: ROIBox | None = None
        self._previous_label: str | None = None
        self._missing_preferred_frames = 0
        self._lock_target, self._lock_target_hand = self._normalize_lock_target(
            lock_target
        )
        self.locked_roi: ROIBox | None = None
        self.locked_center: ROICenter | None = None
        self.locked_label: str | None = None
        self.frames_since_lock_seen = 0
        self.frames_without_any_hand = 0
        self.max_lock_missing_frames = max(0, int(lock_max_missing_frames))
        self.max_no_hand_fallback_frames = max(10, self.max_lock_missing_frames * 2)
        self._lock_max_center_distance_ratio = max(
            0.0,
            float(lock_max_center_distance_ratio),
        )
        self._lock_strict = bool(lock_strict)
        self.last_roi_label: str | None = None
        self._backend: Literal["tasks", "solutions"] = self._resolve_backend()

        if self._backend == "tasks":
            from app.services.hand_detection_service import create_hands_detector

            self._hands = create_hands_detector()
        else:
            self._hands = mp.solutions.hands.Hands(
                static_image_mode=False,
                max_num_hands=2,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5,
            )

    def _resolve_backend(self) -> Literal["tasks", "solutions"]:
        if hasattr(mp, "tasks"):
            return "tasks"
        if hasattr(mp, "solutions") and hasattr(mp.solutions, "hands"):
            return "solutions"
        raise RuntimeError(
            "Installed mediapipe package does not expose mp.tasks or mp.solutions.hands."
        )

    def _normalize_hand_preference(self, value: str) -> ROIHandPreference:
        normalized = str(value or "right").strip().lower()
        if normalized not in {"right", "left", "largest", "first"}:
            raise ValueError(
                "roi_hand_preference must be one of: right, left, largest, first."
            )
        return normalized  # type: ignore[return-value]

    def _normalize_lock_target(self, value: bool | str) -> tuple[bool, ROILockTarget]:
        if isinstance(value, bool):
            return value, self._hand_preference

        normalized = str(value or "right").strip().lower()
        if normalized in {"false", "off", "none", "disabled"}:
            return False, self._hand_preference
        if normalized not in {"right", "left", "largest", "first"}:
            raise ValueError(
                "roi_lock_target must be one of: right, left, largest, first."
            )
        return True, normalized  # type: ignore[return-value]

    def _extract_tasks_label(self, results: object, hand_index: int) -> str | None:
        handedness_list = getattr(results, "handedness", None) or []
        if hand_index >= len(handedness_list):
            return None

        classifications = handedness_list[hand_index] or []
        if not classifications:
            return None

        label = getattr(classifications[0], "category_name", None)
        return str(label).lower() if label else None

    def _extract_solutions_label(self, results: object, hand_index: int) -> str | None:
        handedness_list = getattr(results, "multi_handedness", None) or []
        if hand_index >= len(handedness_list):
            return None

        classification_list = getattr(handedness_list[hand_index], "classification", None)
        if not classification_list:
            return None

        label = getattr(classification_list[0], "label", None)
        return str(label).lower() if label else None

    def _detect_landmarks(self, frame_bgr: np.ndarray) -> list[tuple[object, str | None]]:
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        if self._backend == "tasks":
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
            results = self._hands.detect(mp_image)
            hand_landmarks_list = list(getattr(results, "hand_landmarks", None) or [])
            return [
                (hand_landmarks, self._extract_tasks_label(results, hand_index))
                for hand_index, hand_landmarks in enumerate(hand_landmarks_list)
            ]

        results = self._hands.process(frame_rgb)
        hand_landmarks_list = list(getattr(results, "multi_hand_landmarks", None) or [])
        return [
            (hand_landmarks, self._extract_solutions_label(results, hand_index))
            for hand_index, hand_landmarks in enumerate(hand_landmarks_list)
        ]

    def _build_hand_rois(self, frame_bgr: np.ndarray) -> list[_DetectedHandROI]:
        height, width = frame_bgr.shape[:2]
        detections = self._detect_landmarks(frame_bgr)
        hand_rois: list[_DetectedHandROI] = []

        for hand_landmarks, label in detections:
            landmark_list = getattr(hand_landmarks, "landmark", None)
            if landmark_list is None and isinstance(hand_landmarks, list):
                landmark_list = hand_landmarks

            xs = [float(landmark.x) * width for landmark in landmark_list or []]
            ys = [float(landmark.y) * height for landmark in landmark_list or []]
            if not xs:
                continue

            x1 = int(max(0, min(xs) - self._padding_px))
            y1 = int(max(0, min(ys) - self._padding_px))
            x2 = int(min(width, max(xs) + self._padding_px))
            y2 = int(min(height, max(ys) + self._padding_px))
            if x2 <= x1 or y2 <= y1:
                continue

            hand_rois.append(_DetectedHandROI(roi=(x1, y1, x2, y2), label=label))

        return hand_rois

    def _select_detected_roi(
        self,
        hand_rois: list[_DetectedHandROI],
    ) -> tuple[ROIBox | None, str | None, bool]:
        if not hand_rois:
            if (
                self._previous_roi is not None
                and self._missing_preferred_frames < self._max_missing_frames
            ):
                return (
                    self._previous_roi,
                    self._previous_label or self._hand_preference,
                    False,
                )
            return None, None, False

        if self._hand_preference == "first":
            selected = hand_rois[0]
            return selected.roi, "first", True

        largest = max(hand_rois, key=lambda hand: hand.area)
        if self._hand_preference == "largest":
            return largest.roi, "largest", True

        preferred = next(
            (
                hand
                for hand in hand_rois
                if hand.label == self._hand_preference
            ),
            None,
        )
        if preferred is not None:
            return preferred.roi, self._hand_preference, True

        if (
            self._previous_roi is not None
            and self._missing_preferred_frames < self._max_missing_frames
        ):
            return self._previous_roi, self._previous_label or self._hand_preference, False

        return largest.roi, "largest", True

    def _select_initial_hand_roi(
        self,
        hand_rois: list[_DetectedHandROI],
    ) -> tuple[_DetectedHandROI | None, str | None]:
        if not hand_rois:
            return None, None

        if self._hand_preference == "first":
            return hand_rois[0], "first"

        largest = max(hand_rois, key=lambda hand: hand.area)
        if self._hand_preference == "largest":
            return largest, "largest"

        preferred = next(
            (
                hand
                for hand in hand_rois
                if hand.label == self._hand_preference
            ),
            None,
        )
        if preferred is not None:
            return preferred, self._hand_preference

        return largest, "largest"

    def _lock_target_candidates(
        self,
        hand_rois: list[_DetectedHandROI],
    ) -> list[_DetectedHandROI]:
        if not hand_rois:
            return []

        if self._lock_target_hand == "first":
            return [hand_rois[0]]
        if self._lock_target_hand == "largest":
            return [max(hand_rois, key=lambda hand: hand.area)]

        return [hand for hand in hand_rois if hand.label == self._lock_target_hand]

    def _select_initial_locked_roi(
        self,
        hand_rois: list[_DetectedHandROI],
    ) -> tuple[_DetectedHandROI | None, str | None]:
        target_candidates = self._lock_target_candidates(hand_rois)
        if target_candidates:
            selected = max(target_candidates, key=lambda hand: hand.area)
            return selected, self._lock_target_hand

        # In strict mode, do not bootstrap the lock from the wrong hand.
        if self._lock_strict and self._lock_target_hand in {"right", "left"}:
            return None, None

        return self._select_initial_hand_roi(hand_rois)

    def _update_lock(self, selected: _DetectedHandROI, label: str | None) -> None:
        self.locked_roi = selected.roi
        self.locked_center = selected.center
        self.locked_label = label or selected.label or self._lock_target_hand
        self.frames_since_lock_seen = 0
        self.frames_without_any_hand = 0

    def _select_locked_roi(
        self,
        hand_rois: list[_DetectedHandROI],
        frame_width: int,
    ) -> tuple[ROIBox | None, str | None, bool]:
        if self.locked_roi is None or self.locked_center is None:
            selected, label = self._select_initial_locked_roi(hand_rois)
            if selected is None:
                return None, None, False

            self._update_lock(selected, label)
            return selected.roi, self.locked_label, True

        if hand_rois:
            self.frames_without_any_hand = 0
            target_candidates = self._lock_target_candidates(hand_rois)
            if not target_candidates:
                self.frames_since_lock_seen += 1
                if self._lock_strict:
                    return self.locked_roi, "locked", False

                if self.frames_since_lock_seen <= self.max_no_hand_fallback_frames:
                    return self.locked_roi, "locked", False

                selected, label = self._select_initial_hand_roi(hand_rois)
                if selected is None:
                    return self.locked_roi, "locked", False

                self._update_lock(selected, label)
                return selected.roi, self.locked_label, True

            closest = min(
                target_candidates,
                key=lambda hand: self._center_distance(hand.center, self.locked_center),
            )
            max_distance = frame_width * self._lock_max_center_distance_ratio
            distance = self._center_distance(closest.center, self.locked_center)
            if distance <= max_distance:
                self._update_lock(closest, self.locked_label)
                return closest.roi, self.locked_label, True

            self.frames_since_lock_seen += 1
            return self.locked_roi, "locked", False

        self.frames_without_any_hand += 1
        self.frames_since_lock_seen += 1
        if self.frames_without_any_hand <= self.max_no_hand_fallback_frames:
            return self.locked_roi, "locked", False

        if self._lock_strict:
            return None, None, False

        selected, label = self._select_initial_hand_roi(hand_rois)
        if selected is None:
            return None, None, False

        self._update_lock(selected, label)
        return selected.roi, self.locked_label, True

    def _center_distance(self, center_a: ROICenter, center_b: ROICenter) -> float:
        ax, ay = center_a
        bx, by = center_b
        return float(((ax - bx) ** 2 + (ay - by) ** 2) ** 0.5)

    def detect(self, frame_bgr: np.ndarray) -> Optional[ROIBox]:
        """
        Return a padded hand bounding box for a BGR frame.

        Returns ``None`` when no hand landmarks are detected, allowing callers
        to fall back to full-frame optical flow for that frame.
        """
        self.last_roi_label = None
        hand_rois = self._build_hand_rois(frame_bgr)
        if self._lock_target:
            frame_width = int(frame_bgr.shape[1])
            selected_roi, selected_label, detected_preferred = self._select_locked_roi(
                hand_rois,
                frame_width=frame_width,
            )
        else:
            selected_roi, selected_label, detected_preferred = self._select_detected_roi(
                hand_rois
            )
        if selected_roi is None:
            self._missing_preferred_frames += 1
            return None

        self.last_roi_label = selected_label
        if detected_preferred or selected_label == "largest":
            self._previous_roi = selected_roi
            self._previous_label = selected_label
            self._missing_preferred_frames = 0
        else:
            self._missing_preferred_frames += 1

        return selected_roi

    def close(self) -> None:
        close = getattr(self._hands, "close", None)
        if callable(close):
            close()

    def __enter__(self) -> HandROIDetector:
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()


def crop_to_roi(frame: np.ndarray, roi: ROIBox) -> np.ndarray:
    x1, y1, x2, y2 = roi
    return frame[y1:y2, x1:x2]


def embed_roi_flow_in_canvas(
    roi_flow: np.ndarray,
    roi: ROIBox,
    height: int,
    width: int,
) -> np.ndarray:
    """
    Paste ROI flow back into a zero full-frame flow canvas.

    This keeps downstream feature extraction and visualization shape-compatible
    with the full-frame path while suppressing motion outside the hand ROI.
    """
    x1, y1, x2, y2 = roi
    canvas = np.zeros((height, width, 2), dtype=roi_flow.dtype)
    paste_h = min(y2 - y1, roi_flow.shape[0])
    paste_w = min(x2 - x1, roi_flow.shape[1])
    canvas[y1 : y1 + paste_h, x1 : x1 + paste_w] = roi_flow[:paste_h, :paste_w]
    return canvas
