"""Dynamic Time Warping for angle sequence alignment.

Core DTW logic copied from yolo_scissors_test/backend/app/services/dtw_service.py.
Payload builders and helpers copied from yolo_scissors_test/app/main.py.
"""

from __future__ import annotations

import statistics
from typing import Any


# ── Core DTW ──────────────────────────────────────────────────────────────────

def angle_distance(a: float, b: float) -> float:
    diff = abs(a - b)
    return min(diff, 180 - diff)


def run_dtw(
    expert_angles: list[float],
    learner_angles: list[float],
    window_ratio: float = 0.1,
) -> dict:
    n = len(expert_angles)
    m = len(learner_angles)

    window = max(1, int(max(n, m) * window_ratio))

    dp = [[float("inf")] * (m + 1) for _ in range(n + 1)]
    dp[0][0] = 0

    for i in range(1, n + 1):
        for j in range(max(1, i - window), min(m + 1, i + window)):
            cost = angle_distance(expert_angles[i - 1], learner_angles[j - 1])
            dp[i][j] = cost + min(
                dp[i - 1][j],
                dp[i][j - 1],
                dp[i - 1][j - 1],
            )

    # Backtrack
    i, j = n, m
    path = []

    while i > 0 and j > 0:
        path.append((i - 1, j - 1))

        directions = [
            (dp[i - 1][j], i - 1, j),
            (dp[i][j - 1], i, j - 1),
            (dp[i - 1][j - 1], i - 1, j - 1),
        ]

        _, i, j = min(directions, key=lambda x: x[0])

    path.reverse()

    matches = []
    for i, j in path:
        diff = angle_distance(expert_angles[i], learner_angles[j])
        matches.append(
            {
                "expert_index": i,
                "learner_index": j,
                "expert_angle": expert_angles[i],
                "learner_angle": learner_angles[j],
                "angle_difference": diff,
            }
        )

    total_cost = dp[n][m]
    normalized = total_cost / max(len(path), 1)

    return {
        "dtw_distance": total_cost,
        "normalized_distance": normalized,
        "path": path,
        "matches": matches,
    }


def dtw_window_ratio(expert_angles: list[float], learner_angles: list[float]) -> float:
    max_length = max(len(expert_angles), len(learner_angles), 1)
    required_window_ratio = (abs(len(expert_angles) - len(learner_angles)) + 1) / max_length
    return min(1.0, max(0.1, required_window_ratio + 0.05))


def angle_difference_degrees(angle_a: float, angle_b: float) -> float:
    angle_diff = abs(angle_a - angle_b) % 180
    if angle_diff > 90:
        angle_diff = 180 - angle_diff
    return abs(angle_diff)


# ── Payload builders ──────────────────────────────────────────────────────────

def add_frame_indices_to_dtw_matches(
    dtw_result: dict,
    *,
    expert_frames: list[dict],
    learner_frames: list[dict],
) -> dict:
    enriched_matches = []
    for match in dtw_result["matches"]:
        expert_frame = expert_frames[match["expert_index"]]
        learner_frame = learner_frames[match["learner_index"]]
        enriched_matches.append(
            {
                **match,
                "expert_frame_index": expert_frame["frame_index"],
                "learner_frame_index": learner_frame["frame_index"],
                "expert_timestamp_sec": expert_frame.get("timestamp_sec", 0.0),
                "learner_timestamp_sec": learner_frame.get("timestamp_sec", 0.0),
                "angle_difference": round(
                    angle_difference_degrees(
                        float(match["expert_angle"]),
                        float(match["learner_angle"]),
                    ),
                    6,
                ),
            }
        )

    return {
        **dtw_result,
        "matches": enriched_matches,
    }


def build_dtw_alignment_payload(
    dtw_result: dict,
    *,
    expert_valid_frame_count: int,
    learner_valid_frame_count: int,
) -> dict:
    matches = []
    for step, match in enumerate(dtw_result.get("matches", [])):
        matches.append(
            {
                "dtw_step": step,
                "expert_seq_index": int(match["expert_index"]),
                "learner_seq_index": int(match["learner_index"]),
                "expert_frame_index": int(match["expert_frame_index"]),
                "learner_frame_index": int(match["learner_frame_index"]),
                "expert_timestamp_sec": round(float(match.get("expert_timestamp_sec", 0.0)), 6),
                "learner_timestamp_sec": round(float(match.get("learner_timestamp_sec", 0.0)), 6),
                "expert_angle": round(float(match["expert_angle"]), 6),
                "learner_angle": round(float(match["learner_angle"]), 6),
                "angle_difference": round(float(match["angle_difference"]), 6),
            }
        )

    return {
        "dtw_distance": round(float(dtw_result["dtw_distance"]), 6),
        "normalized_distance": round(float(dtw_result["normalized_distance"]), 6),
        "num_matches": len(matches),
        "expert_valid_frame_count": expert_valid_frame_count,
        "learner_valid_frame_count": learner_valid_frame_count,
        "matches": matches,
    }


def build_learner_comparison_payload(
    *,
    learner_frames_payload: dict,
    dtw_alignment_payload: dict,
) -> dict:
    direct_matches: dict[int, list[dict]] = {}
    for match in dtw_alignment_payload["matches"]:
        learner_frame_index = int(match["learner_frame_index"])
        direct_matches.setdefault(learner_frame_index, []).append(
            {
                "expert_frame_index": int(match["expert_frame_index"]),
                "expert_timestamp_sec": float(match["expert_timestamp_sec"]),
                "expert_angle": float(match["expert_angle"]),
            }
        )

    direct_frame_indices = sorted(direct_matches)
    frames = []
    for learner_frame in learner_frames_payload["frames"]:
        frame_index = int(learner_frame["frame_index"])
        learner_angle = learner_frame.get("line_angle")
        matched = _resolve_learner_match(
            frame_index=frame_index,
            direct_matches=direct_matches,
            direct_frame_indices=direct_frame_indices,
        )

        matched_angle = matched.get("matched_expert_angle")
        angle_diff = (
            round(angle_difference_degrees(float(learner_angle), float(matched_angle)), 6)
            if learner_angle is not None and matched_angle is not None
            else None
        )

        frames.append(
            {
                "frame_index": frame_index,
                "timestamp_sec": learner_frame["timestamp_sec"],
                "learner_detected": learner_frame["detected"],
                "learner_valid_line": learner_frame["valid_line"],
                "learner_angle": learner_angle,
                "learner_line_center": learner_frame["line_center"],
                "matched_expert_frame_index": matched.get("matched_expert_frame_index"),
                "matched_expert_timestamp_sec": matched.get("matched_expert_timestamp_sec"),
                "matched_expert_angle": matched_angle,
                "match_source": matched.get("match_source"),
                "angle_difference": angle_diff,
                "error_level": _error_level(angle_diff),
            }
        )

    return {
        "video_type": "learner_comparison",
        "reference_mode": "dynamic_dtw_expert_angle",
        "frames": frames,
    }


def build_run_summary_payload(
    *,
    clip_id: str,
    expert_name: str,
    expert_frames_payload: dict,
    learner_frames_payload: dict,
    dtw_alignment_payload: dict,
    learner_comparison_payload: dict,
) -> dict:
    expert_total_frames = int(expert_frames_payload["total_frames"])
    learner_total_frames = int(learner_frames_payload["total_frames"])
    expert_valid_lines = sum(1 for frame in expert_frames_payload["frames"] if frame["valid_line"])
    learner_valid_lines = sum(1 for frame in learner_frames_payload["frames"] if frame["valid_line"])
    angle_differences = [
        float(frame["angle_difference"])
        for frame in learner_comparison_payload["frames"]
        if frame.get("angle_difference") is not None
    ]

    return {
        "clip_id": clip_id,
        "expert_name": expert_name,
        "expert_total_frames": expert_total_frames,
        "learner_total_frames": learner_total_frames,
        "expert_valid_lines": expert_valid_lines,
        "learner_valid_lines": learner_valid_lines,
        "expert_valid_ratio": round(expert_valid_lines / max(expert_total_frames, 1), 6),
        "learner_valid_ratio": round(learner_valid_lines / max(learner_total_frames, 1), 6),
        "dtw_distance": dtw_alignment_payload["dtw_distance"],
        "normalized_dtw_distance": dtw_alignment_payload["normalized_distance"],
        "num_dtw_matches": dtw_alignment_payload["num_matches"],
        "mean_angle_difference": (
            round(float(statistics.mean(angle_differences)), 6)
            if angle_differences
            else None
        ),
        "median_angle_difference": (
            round(float(statistics.median(angle_differences)), 6)
            if angle_differences
            else None
        ),
        "max_angle_difference": (
            round(max(angle_differences), 6) if angle_differences else None
        ),
        "high_error_frame_count": sum(
            1
            for frame in learner_comparison_payload["frames"]
            if frame["error_level"] == "high"
        ),
        "medium_error_frame_count": sum(
            1
            for frame in learner_comparison_payload["frames"]
            if frame["error_level"] == "medium"
        ),
        "ok_frame_count": sum(
            1
            for frame in learner_comparison_payload["frames"]
            if frame["error_level"] == "ok"
        ),
    }


# ── Private helpers ────────────────────────────────────────────────────────────

def _resolve_learner_match(
    *,
    frame_index: int,
    direct_matches: dict[int, list[dict]],
    direct_frame_indices: list[int],
) -> dict:
    if not direct_frame_indices:
        return {
            "matched_expert_frame_index": None,
            "matched_expert_timestamp_sec": None,
            "matched_expert_angle": None,
            "match_source": None,
        }

    match_source = "dtw_direct"
    source_frame_index = frame_index
    matches = direct_matches.get(frame_index)

    if matches is None:
        previous_indices = [idx for idx in direct_frame_indices if idx < frame_index]
        if previous_indices:
            source_frame_index = previous_indices[-1]
            matches = direct_matches[source_frame_index]
            match_source = "nearest_previous"
        else:
            next_indices = [idx for idx in direct_frame_indices if idx > frame_index]
            if not next_indices:
                return {
                    "matched_expert_frame_index": None,
                    "matched_expert_timestamp_sec": None,
                    "matched_expert_angle": None,
                    "match_source": None,
                }
            source_frame_index = next_indices[0]
            matches = direct_matches[source_frame_index]
            match_source = "nearest_next"

    expert_angles = [float(match["expert_angle"]) for match in matches]
    expert_frame_indices = [int(match["expert_frame_index"]) for match in matches]
    expert_timestamps = [float(match["expert_timestamp_sec"]) for match in matches]

    return {
        "matched_expert_frame_index": int(round(float(statistics.median(expert_frame_indices)))),
        "matched_expert_timestamp_sec": round(float(statistics.median(expert_timestamps)), 6),
        "matched_expert_angle": round(float(statistics.median(expert_angles)), 6),
        "match_source": match_source,
        "source_learner_frame_index": source_frame_index,
    }


def _error_level(angle_difference: float | None) -> str:
    if angle_difference is None:
        return "unknown"
    if angle_difference <= 10:
        return "ok"
    if angle_difference <= 25:
        return "medium"
    return "high"
