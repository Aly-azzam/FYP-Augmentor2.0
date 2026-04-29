"""Helper functions for the MediaPipe Hands extraction pipeline.

These utilities are intentionally small, deterministic, and JSON friendly.
They are inspired by the existing landmark / motion helpers in this repo
(``sequence_utils.smooth_sequence``, ``landmark_cleaning_service`` style
interpolation, ``math_utils.angle_between_vectors``) but rewritten around
the single-selected-hand data model used by this new pipeline.
"""

from __future__ import annotations

import math
from typing import Iterable, List, Optional, Sequence

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# MediaPipe Hands landmark indices and names.
#
# Reference (MediaPipe Hands):
#   0  wrist
#   1  thumb_cmc        2  thumb_mcp        3  thumb_ip        4  thumb_tip
#   5  index_finger_mcp 6  index_finger_pip 7  index_finger_dip 8  index_finger_tip
#   9  middle_finger_mcp 10 middle_finger_pip 11 middle_finger_dip 12 middle_finger_tip
#  13  ring_finger_mcp  14 ring_finger_pip  15 ring_finger_dip  16 ring_finger_tip
#  17  pinky_mcp        18 pinky_pip        19 pinky_dip        20 pinky_tip
# ---------------------------------------------------------------------------

HAND_LANDMARK_NAMES: List[str] = [
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

PALM_CENTER_INDICES: List[int] = [0, 5, 9, 13, 17]

_FLOAT_EPSILON = 1e-9


# ---------------------------------------------------------------------------
# Image / landmark primitives
# ---------------------------------------------------------------------------

def bgr_to_rgb(bgr_image: np.ndarray) -> np.ndarray:
    """Convert an OpenCV BGR frame to RGB for MediaPipe consumption."""
    if bgr_image is None:
        raise ValueError("bgr_to_rgb: bgr_image must not be None.")
    if not isinstance(bgr_image, np.ndarray):
        raise TypeError("bgr_to_rgb: bgr_image must be a numpy.ndarray.")
    return cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)


def landmark_to_list(landmark) -> List[float]:
    """Convert a MediaPipe landmark proto (or dict/iterable) to ``[x, y, z]``.

    Any NaN component is rejected early to avoid silently corrupting the
    downstream JSON document.
    """
    if landmark is None:
        raise ValueError("landmark_to_list: landmark must not be None.")

    if hasattr(landmark, "x") and hasattr(landmark, "y") and hasattr(landmark, "z"):
        x, y, z = float(landmark.x), float(landmark.y), float(landmark.z)
    elif isinstance(landmark, dict):
        x = float(landmark.get("x", 0.0))
        y = float(landmark.get("y", 0.0))
        z = float(landmark.get("z", 0.0))
    else:
        values = list(landmark)
        if len(values) < 2:
            raise ValueError("landmark_to_list: iterable must contain at least x and y.")
        x = float(values[0])
        y = float(values[1])
        z = float(values[2]) if len(values) >= 3 else 0.0

    if any(math.isnan(v) for v in (x, y, z)):
        raise ValueError("landmark_to_list: landmark contains NaN.")
    return [x, y, z]


def normalize_handedness_label(label: Optional[str]) -> str:
    """Map a raw MediaPipe handedness label into ``Left`` / ``Right`` / ``Unknown``."""
    if label is None:
        return "Unknown"
    normalized = str(label).strip().lower()
    if normalized in {"left", "l"}:
        return "Left"
    if normalized in {"right", "r"}:
        return "Right"
    return "Unknown"


# ---------------------------------------------------------------------------
# Hand geometry
# ---------------------------------------------------------------------------

def compute_hand_center(landmarks: Sequence[Sequence[float]]) -> List[float]:
    """Return the 3D palm center ``[x, y, z]`` from the palm landmark indices."""
    if len(landmarks) < 21:
        raise ValueError("compute_hand_center: expected 21 landmarks.")
    xs = [float(landmarks[i][0]) for i in PALM_CENTER_INDICES]
    ys = [float(landmarks[i][1]) for i in PALM_CENTER_INDICES]
    zs = [float(landmarks[i][2]) for i in PALM_CENTER_INDICES]
    return [sum(xs) / len(xs), sum(ys) / len(ys), sum(zs) / len(zs)]


def compute_bbox_from_landmarks(landmarks: Sequence[Sequence[float]]) -> dict:
    """Compute an axis-aligned bounding box around all 21 landmarks."""
    if not landmarks:
        raise ValueError("compute_bbox_from_landmarks: empty landmark list.")
    xs = [float(pt[0]) for pt in landmarks]
    ys = [float(pt[1]) for pt in landmarks]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    return {
        "x_min": x_min,
        "y_min": y_min,
        "x_max": x_max,
        "y_max": y_max,
        "width": x_max - x_min,
        "height": y_max - y_min,
    }


def normalize_point_to_box(
    point: Sequence[float],
    bbox: dict,
) -> List[float]:
    """Remap a point ``[x, y, (z)]`` into bbox-relative coordinates in [0, 1]."""
    width = float(bbox.get("width", 0.0))
    height = float(bbox.get("height", 0.0))
    x_min = float(bbox.get("x_min", 0.0))
    y_min = float(bbox.get("y_min", 0.0))

    safe_width = width if width > _FLOAT_EPSILON else 1.0
    safe_height = height if height > _FLOAT_EPSILON else 1.0

    nx = (float(point[0]) - x_min) / safe_width
    ny = (float(point[1]) - y_min) / safe_height
    if len(point) >= 3:
        return [nx, ny, float(point[2])]
    return [nx, ny, 0.0]


# ---------------------------------------------------------------------------
# Angles / orientation
# ---------------------------------------------------------------------------

def compute_joint_angle(
    point_a: Sequence[float],
    point_b: Sequence[float],
    point_c: Sequence[float],
) -> float:
    """Angle (in degrees) at joint ``b`` formed by the vectors ``ba`` and ``bc``."""
    a = np.asarray(point_a, dtype=float)
    b = np.asarray(point_b, dtype=float)
    c = np.asarray(point_c, dtype=float)
    vector_ba = a - b
    vector_bc = c - b
    norm_ba = float(np.linalg.norm(vector_ba))
    norm_bc = float(np.linalg.norm(vector_bc))
    if norm_ba < _FLOAT_EPSILON or norm_bc < _FLOAT_EPSILON:
        return 0.0
    cos_angle = float(np.dot(vector_ba, vector_bc) / (norm_ba * norm_bc))
    cos_angle = max(-1.0, min(1.0, cos_angle))
    return math.degrees(math.acos(cos_angle))


def compute_hand_orientation(landmarks: Sequence[Sequence[float]]) -> float:
    """Return the angle (degrees) of the wrist -> middle-finger-MCP vector.

    The angle is measured in image space with 0° pointing along ``+x`` and
    increasing counter-clockwise. Note: in image coordinates ``y`` grows
    downward, so we negate ``dy`` so that the reported angle matches the
    intuitive screen-space orientation.
    """
    if len(landmarks) < 21:
        raise ValueError("compute_hand_orientation: expected 21 landmarks.")
    wrist = landmarks[0]
    middle_mcp = landmarks[9]
    dx = float(middle_mcp[0]) - float(wrist[0])
    dy = float(middle_mcp[1]) - float(wrist[1])
    if abs(dx) < _FLOAT_EPSILON and abs(dy) < _FLOAT_EPSILON:
        return 0.0
    return math.degrees(math.atan2(-dy, dx))


# ---------------------------------------------------------------------------
# Temporal helpers
# ---------------------------------------------------------------------------

def smooth_sequence(values: Sequence[float], window: int = 3) -> List[float]:
    """Centered moving-average smoother.

    Mirrors ``app.utils.sequence_utils.smooth_sequence`` but accepts any
    ``Sequence[float]`` and returns a plain ``list[float]``.
    """
    values_list = [float(v) for v in values]
    if window <= 1 or len(values_list) <= 1:
        return values_list
    if len(values_list) <= window:
        # Fall back to a simple full-window mean applied uniformly.
        mean_value = sum(values_list) / len(values_list)
        return [mean_value for _ in values_list]

    result: List[float] = []
    half = window // 2
    for i in range(len(values_list)):
        start = max(0, i - half)
        end = min(len(values_list), i + half + 1)
        window_slice = values_list[start:end]
        result.append(sum(window_slice) / len(window_slice))
    return result


def interpolate_missing_points(
    sequence: Sequence[Optional[Sequence[float]]],
    max_gap: int = 2,
) -> List[Optional[List[float]]]:
    """Linearly interpolate short gaps of ``None`` entries in a vector sequence.

    Each element is expected to be either ``None`` or a vector of equal
    dimensionality (typically ``[x, y]`` or ``[x, y, z]``). Gaps longer than
    ``max_gap`` are left as ``None``. Leading and trailing ``None`` runs are
    always left untouched.
    """
    if max_gap < 0:
        raise ValueError("interpolate_missing_points: max_gap must be >= 0.")

    output: List[Optional[List[float]]] = [
        list(map(float, point)) if point is not None else None for point in sequence
    ]

    index = 0
    while index < len(output):
        if output[index] is not None:
            index += 1
            continue

        gap_start = index
        while index < len(output) and output[index] is None:
            index += 1
        gap_end = index - 1
        gap_length = gap_end - gap_start + 1

        prev_index = gap_start - 1
        next_index = index if index < len(output) else None

        if (
            gap_length <= max_gap
            and prev_index >= 0
            and next_index is not None
            and output[prev_index] is not None
            and output[next_index] is not None
        ):
            start_point = output[prev_index]
            end_point = output[next_index]
            assert start_point is not None and end_point is not None
            if len(start_point) != len(end_point):
                raise ValueError(
                    "interpolate_missing_points: mismatched vector dimensions across gap."
                )
            for offset in range(1, gap_length + 1):
                alpha = offset / (gap_length + 1)
                interpolated = [
                    (1.0 - alpha) * start_point[dim] + alpha * end_point[dim]
                    for dim in range(len(start_point))
                ]
                output[gap_start + offset - 1] = interpolated
    return output


def compute_velocity(
    current: Optional[Sequence[float]],
    previous: Optional[Sequence[float]],
    dt: float,
) -> Optional[List[float]]:
    """Per-component velocity vector ``(current - previous) / dt``.

    Returns ``None`` if either point is missing or ``dt`` is non-positive.
    """
    if current is None or previous is None or dt <= _FLOAT_EPSILON:
        return None
    if len(current) != len(previous):
        raise ValueError("compute_velocity: current and previous must share dimensionality.")
    return [(float(current[i]) - float(previous[i])) / float(dt) for i in range(len(current))]


def iter_float_triplets(vectors: Iterable[Sequence[float]]) -> List[List[float]]:
    """Utility to deep-copy a sequence of float triplets into plain lists."""
    return [[float(v[0]), float(v[1]), float(v[2])] for v in vectors]
