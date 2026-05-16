from __future__ import annotations
from .schemas import VideoFlowSummary, VibrationAnalysis, AffectedWindow


_MIN_CONSECUTIVE_WINDOWS = 3
_MIN_FREQ_HZ = 4.0
_FREQ_TOLERANCE_HZ = 2.0


def _get_severity(consecutive: int, peak_confidence: float) -> str:
    if consecutive >= 4 or peak_confidence > 0.30:
        return "severe"
    if peak_confidence >= 0.20:
        return "moderate"
    return "mild"


def classify_vibration(summary: VideoFlowSummary) -> VibrationAnalysis:
    """
    Apply rule-based vibration detection on a computed VideoFlowSummary.

    Rules:
      1. max_consecutive_windows >= 3
      2. consistent_freq_hz >= 4.0 Hz

    If both pass, scan vibration_windows to find the exact streak,
    extract timestamps, peak confidence, duration, and severity.
    """
    # Rule check
    if (
        summary.max_consecutive_windows < _MIN_CONSECUTIVE_WINDOWS
        or summary.consistent_freq_hz < _MIN_FREQ_HZ
    ):
        return VibrationAnalysis(vibration_detected=False)

    windows = summary.vibration_windows
    if not windows:
        return VibrationAnalysis(vibration_detected=False)

    # Find the best consecutive streak in vibration_windows
    best_streak: list[dict] = []
    current_streak: list[dict] = []

    for w in windows:
        if w["confidence"] >= 0.15:
            if not current_streak or abs(w["dominant_freq_hz"] - current_streak[-1]["dominant_freq_hz"]) <= _FREQ_TOLERANCE_HZ:
                current_streak.append(w)
            else:
                if len(current_streak) > len(best_streak):
                    best_streak = current_streak
                current_streak = [w]
        else:
            if len(current_streak) > len(best_streak):
                best_streak = current_streak
            current_streak = []

    if len(current_streak) > len(best_streak):
        best_streak = current_streak

    if len(best_streak) < _MIN_CONSECUTIVE_WINDOWS:
        return VibrationAnalysis(vibration_detected=False)

    # Extract results from best streak
    peak_window = max(best_streak, key=lambda w: w["confidence"])
    peak_timestamp = peak_window["timestamp_sec"]
    peak_freq = peak_window["dominant_freq_hz"]
    peak_confidence = peak_window["confidence"]
    first_timestamp = best_streak[0]["timestamp_sec"]
    last_timestamp = best_streak[-1]["timestamp_sec"]
    duration = round(last_timestamp - first_timestamp + 1.0, 2)

    affected = [
        AffectedWindow(
            timestamp_sec=w["timestamp_sec"],
            confidence=w["confidence"],
            freq_hz=w["dominant_freq_hz"],
        )
        for w in best_streak
    ]

    severity = _get_severity(len(best_streak), peak_confidence)

    return VibrationAnalysis(
        vibration_detected=True,
        severity=severity,
        peak_timestamp_sec=peak_timestamp,
        peak_freq_hz=peak_freq,
        duration_sec=duration,
        affected_windows=affected,
    )
