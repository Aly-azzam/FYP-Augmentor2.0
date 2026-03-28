"""Context-validation gate that rejects clearly out-of-context learner videos.

Runs BEFORE DTW / metrics / scoring.  Uses only deterministic signals already
available from the MotionRepresentationOutput (MediaPipe landmarks, velocities,
hand-presence flags, feature vectors).

If the gate rejects, the caller should short-circuit the pipeline and return
score=0, status="out_of_context".
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any


# ── Thresholds (tunable constants) ──────────────────────────────────────────

MIN_HAND_PRESENCE_RATIO = 0.15
MIN_HAND_PRESENCE_RATIO_RELATIVE_TO_EXPERT = 0.45
MIN_USABLE_FRAMES = 5
MIN_MOTION_ENERGY = 0.002
MAX_MOTION_ENERGY_RATIO_DIFF = 10.0
MAX_WORKSPACE_CENTER_DRIFT = 0.55
MIN_HAND_SCALE_RATIO_TO_EXPERT = 0.35
MIN_ACTIVITY_PATTERN_SIMILARITY = 0.55
MIN_DTW_SIMILARITY_FOR_CONTEXT = 0.10


@dataclass
class GateSignals:
    """Collected signals used by the gate decision."""
    expert_total_frames: int = 0
    learner_total_frames: int = 0
    expert_valid_frames: int = 0
    learner_valid_frames: int = 0
    expert_hand_presence_ratio: float = 0.0
    learner_hand_presence_ratio: float = 0.0
    expert_usable_frames: int = 0
    learner_usable_frames: int = 0
    expert_motion_energy: float = 0.0
    learner_motion_energy: float = 0.0
    motion_energy_ratio: float = 0.0
    expert_workspace_center: list[float] = field(default_factory=lambda: [0.0, 0.0])
    learner_workspace_center: list[float] = field(default_factory=lambda: [0.0, 0.0])
    workspace_center_distance: float = 0.0
    expert_mean_hand_scale: float = 0.0
    learner_mean_hand_scale: float = 0.0
    hand_scale_ratio: float = 0.0
    activity_pattern_similarity: float = 1.0
    dtw_similarity: float = 1.0


@dataclass
class GateResult:
    """Output of the context gate."""
    passed: bool
    reasons: list[str] = field(default_factory=list)
    signals: GateSignals = field(default_factory=GateSignals)


def run_context_gate(
    expert_motion: dict[str, Any],
    learner_motion: dict[str, Any],
    dtw_similarity: float | None = None,
) -> GateResult:
    """Evaluate whether the learner video is plausibly the same task as the expert."""

    signals = GateSignals()
    reasons: list[str] = []

    expert_frames = expert_motion.get("frames", [])
    learner_frames = learner_motion.get("frames", [])

    signals.expert_total_frames = len(expert_frames)
    signals.learner_total_frames = len(learner_frames)

    signals.expert_valid_frames = _count_present(expert_frames)
    signals.learner_valid_frames = _count_present(learner_frames)

    signals.expert_hand_presence_ratio = (
        signals.expert_valid_frames / max(signals.expert_total_frames, 1)
    )
    signals.learner_hand_presence_ratio = (
        signals.learner_valid_frames / max(signals.learner_total_frames, 1)
    )

    signals.expert_usable_frames = signals.expert_valid_frames
    signals.learner_usable_frames = signals.learner_valid_frames

    signals.expert_motion_energy = _compute_motion_energy(expert_frames)
    signals.learner_motion_energy = _compute_motion_energy(learner_frames)
    if signals.expert_motion_energy > 0:
        signals.motion_energy_ratio = signals.learner_motion_energy / signals.expert_motion_energy
    else:
        signals.motion_energy_ratio = 0.0

    signals.expert_workspace_center = _compute_workspace_center(expert_frames)
    signals.learner_workspace_center = _compute_workspace_center(learner_frames)
    signals.workspace_center_distance = _euclidean_2d(
        signals.expert_workspace_center, signals.learner_workspace_center
    )
    signals.expert_mean_hand_scale = _compute_mean_hand_scale(expert_frames)
    signals.learner_mean_hand_scale = _compute_mean_hand_scale(learner_frames)
    if signals.expert_mean_hand_scale > 0.0:
        signals.hand_scale_ratio = signals.learner_mean_hand_scale / signals.expert_mean_hand_scale
    else:
        signals.hand_scale_ratio = 0.0
    signals.activity_pattern_similarity = _compute_activity_pattern_similarity(
        expert_frames=expert_frames,
        learner_frames=learner_frames,
    )

    if dtw_similarity is not None:
        signals.dtw_similarity = dtw_similarity

    # ── Gate checks (order: cheapest / most obvious first) ──────────────

    if signals.learner_hand_presence_ratio < MIN_HAND_PRESENCE_RATIO:
        reasons.append(
            f"Hand presence too low ({signals.learner_hand_presence_ratio:.1%} < {MIN_HAND_PRESENCE_RATIO:.0%}). "
            "No hands detected in most of the learner video."
        )
    required_presence_from_expert = max(
        MIN_HAND_PRESENCE_RATIO,
        signals.expert_hand_presence_ratio * MIN_HAND_PRESENCE_RATIO_RELATIVE_TO_EXPERT,
    )
    if signals.learner_hand_presence_ratio < required_presence_from_expert:
        reasons.append(
            f"Learner hand coverage is much lower than expert "
            f"({signals.learner_hand_presence_ratio:.1%} vs expert {signals.expert_hand_presence_ratio:.1%})."
        )

    if signals.learner_usable_frames < MIN_USABLE_FRAMES:
        reasons.append(
            f"Too few usable frames ({signals.learner_usable_frames} < {MIN_USABLE_FRAMES}). "
            "The video is too short or hands are not visible."
        )

    if signals.learner_motion_energy < MIN_MOTION_ENERGY:
        reasons.append(
            f"Learner motion energy is negligible ({signals.learner_motion_energy:.5f}). "
            "No meaningful hand movement detected."
        )

    if (
        signals.expert_motion_energy > MIN_MOTION_ENERGY
        and signals.learner_motion_energy > MIN_MOTION_ENERGY
    ):
        ratio = signals.motion_energy_ratio
        if ratio > MAX_MOTION_ENERGY_RATIO_DIFF or (ratio > 0 and ratio < 1.0 / MAX_MOTION_ENERGY_RATIO_DIFF):
            reasons.append(
                f"Motion activity level mismatch (ratio={ratio:.2f}). "
                "Expert and learner show very different amounts of hand movement."
            )

    if (
        signals.learner_hand_presence_ratio >= MIN_HAND_PRESENCE_RATIO
        and signals.expert_hand_presence_ratio >= MIN_HAND_PRESENCE_RATIO
        and signals.workspace_center_distance > MAX_WORKSPACE_CENTER_DRIFT
    ):
        reasons.append(
            f"Workspace position mismatch (distance={signals.workspace_center_distance:.3f} > {MAX_WORKSPACE_CENTER_DRIFT}). "
            "Hands are in a very different part of the frame compared to the expert."
        )
    if (
        signals.expert_mean_hand_scale > 0.0
        and signals.learner_mean_hand_scale > 0.0
        and signals.hand_scale_ratio < MIN_HAND_SCALE_RATIO_TO_EXPERT
    ):
        reasons.append(
            f"Detected hand size is much smaller than expected (ratio={signals.hand_scale_ratio:.2f}). "
            "The learner video appears zoomed out / out of task context."
        )
    if signals.activity_pattern_similarity < MIN_ACTIVITY_PATTERN_SIMILARITY:
        reasons.append(
            f"Motion activity pattern mismatch (similarity={signals.activity_pattern_similarity:.2f}). "
            "The learner motion timeline does not resemble the expert task."
        )

    if dtw_similarity is not None and dtw_similarity < MIN_DTW_SIMILARITY_FOR_CONTEXT:
        reasons.append(
            f"Overall motion pattern is extremely different from the expert (DTW similarity={dtw_similarity:.3f}). "
        )

    print(f"[GATE] gate_entered=True")
    print(f"[GATE] signals: hand_presence_ratio={signals.learner_hand_presence_ratio:.3f}, "
          f"usable_frames={signals.learner_usable_frames}, "
          f"motion_energy=({signals.expert_motion_energy:.5f} vs {signals.learner_motion_energy:.5f}), "
          f"energy_ratio={signals.motion_energy_ratio:.3f}, "
          f"workspace_dist={signals.workspace_center_distance:.3f}, "
          f"scale_ratio={signals.hand_scale_ratio:.3f}, "
          f"activity_sim={signals.activity_pattern_similarity:.3f}, "
          f"dtw_sim={signals.dtw_similarity:.3f}")

    if reasons:
        print(f"[GATE] gate_passed=False, reasons={reasons}")
        return GateResult(passed=False, reasons=reasons, signals=signals)

    print("[GATE] gate_passed=True")
    return GateResult(passed=True, reasons=[], signals=signals)


# ── Helpers ─────────────────────────────────────────────────────────────────

def _count_present(frames: list[dict[str, Any]]) -> int:
    count = 0
    for frame in frames:
        left = frame.get("left_hand_features", {})
        right = frame.get("right_hand_features", {})
        if isinstance(left, dict) and left.get("present"):
            count += 1
        elif isinstance(right, dict) and right.get("present"):
            count += 1
    return count


def _compute_motion_energy(frames: list[dict[str, Any]]) -> float:
    """Average velocity magnitude across all present hand frames."""
    magnitudes: list[float] = []
    for frame in frames:
        for hand_key in ("left_hand_features", "right_hand_features"):
            hand = frame.get(hand_key, {})
            if not isinstance(hand, dict) or not hand.get("present"):
                continue
            for vel_key in ("wrist_velocity", "palm_velocity"):
                vel = hand.get(vel_key)
                if isinstance(vel, list) and len(vel) >= 2:
                    mag = math.sqrt(sum(float(v) ** 2 for v in vel[:3] if isinstance(v, (int, float))))
                    magnitudes.append(mag)
    if not magnitudes:
        return 0.0
    return sum(magnitudes) / len(magnitudes)


def _compute_workspace_center(frames: list[dict[str, Any]]) -> list[float]:
    """Average XY position of all detected wrists/palms."""
    xs: list[float] = []
    ys: list[float] = []
    for frame in frames:
        for hand_key in ("left_hand_features", "right_hand_features"):
            hand = frame.get(hand_key, {})
            if not isinstance(hand, dict) or not hand.get("present"):
                continue
            for pos_key in ("wrist_position", "palm_center"):
                pos = hand.get(pos_key)
                if isinstance(pos, list) and len(pos) >= 2:
                    xs.append(float(pos[0]))
                    ys.append(float(pos[1]))
    if not xs:
        return [0.0, 0.0]
    return [sum(xs) / len(xs), sum(ys) / len(ys)]


def _compute_mean_hand_scale(frames: list[dict[str, Any]]) -> float:
    scales: list[float] = []
    for frame in frames:
        for hand_key in ("left_hand_features", "right_hand_features"):
            hand = frame.get(hand_key, {})
            if not isinstance(hand, dict) or not hand.get("present"):
                continue
            scale = hand.get("hand_scale")
            if isinstance(scale, (int, float)):
                scales.append(float(scale))
    if not scales:
        return 0.0
    return sum(scales) / len(scales)


def _compute_activity_pattern_similarity(
    *,
    expert_frames: list[dict[str, Any]],
    learner_frames: list[dict[str, Any]],
) -> float:
    """Return [0,1] similarity between expert and learner motion-intensity timelines."""
    expert_series = _extract_motion_intensity_series(expert_frames)
    learner_series = _extract_motion_intensity_series(learner_frames)
    if not expert_series or not learner_series:
        return 0.0

    target_len = max(24, min(80, max(len(expert_series), len(learner_series))))
    expert_resampled = _resample_series(expert_series, target_len)
    learner_resampled = _resample_series(learner_series, target_len)

    level_sim = _cosine_similarity(expert_resampled, learner_resampled)
    expert_diff = _first_diff(expert_resampled)
    learner_diff = _first_diff(learner_resampled)
    trend_sim = _cosine_similarity(expert_diff, learner_diff)

    # blend absolute level + temporal trend similarity
    blended = (0.6 * level_sim) + (0.4 * trend_sim)
    return max(0.0, min(1.0, blended))


def _extract_motion_intensity_series(frames: list[dict[str, Any]]) -> list[float]:
    series: list[float] = []
    for frame in frames:
        per_frame: list[float] = []
        for hand_key in ("left_hand_features", "right_hand_features"):
            hand = frame.get(hand_key, {})
            if not isinstance(hand, dict) or not hand.get("present"):
                continue
            wrist = hand.get("wrist_velocity")
            palm = hand.get("palm_velocity")
            if isinstance(wrist, list) and len(wrist) >= 2:
                per_frame.append(_vector_magnitude(wrist))
            if isinstance(palm, list) and len(palm) >= 2:
                per_frame.append(_vector_magnitude(palm))
        series.append(sum(per_frame) / len(per_frame) if per_frame else 0.0)
    return series


def _resample_series(series: list[float], target_len: int) -> list[float]:
    if not series:
        return [0.0] * max(1, target_len)
    if len(series) == target_len:
        return [float(v) for v in series]
    if len(series) == 1:
        return [float(series[0])] * target_len

    out: list[float] = []
    max_src_index = len(series) - 1
    for idx in range(target_len):
        pos = (idx / max(target_len - 1, 1)) * max_src_index
        lo = int(math.floor(pos))
        hi = min(lo + 1, max_src_index)
        alpha = pos - lo
        value = (1.0 - alpha) * float(series[lo]) + alpha * float(series[hi])
        out.append(value)
    return out


def _first_diff(series: list[float]) -> list[float]:
    if len(series) < 2:
        return [0.0]
    return [float(series[i]) - float(series[i - 1]) for i in range(1, len(series))]


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    n = min(len(a), len(b))
    if n == 0:
        return 0.0
    aa = [float(x) for x in a[:n]]
    bb = [float(x) for x in b[:n]]

    # center to reduce bias from absolute offset
    a_mean = sum(aa) / n
    b_mean = sum(bb) / n
    aa = [x - a_mean for x in aa]
    bb = [x - b_mean for x in bb]

    dot = sum(x * y for x, y in zip(aa, bb))
    norm_a = math.sqrt(sum(x * x for x in aa))
    norm_b = math.sqrt(sum(y * y for y in bb))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    raw = dot / (norm_a * norm_b)
    # map [-1,1] -> [0,1]
    return max(0.0, min(1.0, (raw + 1.0) * 0.5))


def _vector_magnitude(vector: list[float]) -> float:
    return math.sqrt(sum(float(v) ** 2 for v in vector[:3] if isinstance(v, (int, float))))


def _euclidean_2d(a: list[float], b: list[float]) -> float:
    dx = a[0] - b[0]
    dy = a[1] - b[1]
    return math.sqrt(dx * dx + dy * dy)


OUT_OF_CONTEXT_MESSAGE = (
    "This video does not appear to match the expected activity for this chapter. "
    "Please upload a practice attempt showing the same task as the expert demonstration."
)
