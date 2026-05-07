"""Trajectory error detection — Step 5 of the AugMentor pipeline.

Converts per-frame deviation data from aligned_corridor.json into clean error
events using a state machine with hysteresis and merge rules.

Rules:
  1. Minimum duration: an outside window must span at least
     ``min_outside_original_frames`` original frames (≥1 strided frame) to count.
  2. Merge: two outside windows separated by <1 s inside are merged into one.
  3. State machine: INSIDE→OUTSIDE opens an error; OUTSIDE→INSIDE closes it.

Outputs:
  <run_dir>/trajectory_errors.json
  <run_dir>/trajectory_errors_preview.png
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

import cv2
import numpy as np

logger = logging.getLogger(__name__)

TRAJECTORY_ERRORS_FILENAME = "trajectory_errors.json"
TRAJECTORY_ERRORS_PREVIEW_FILENAME = "trajectory_errors_preview.png"

# Drawing palette (BGR)
_C_CORRIDOR_FILL = (0, 180, 0)
_C_CENTERLINE = (255, 255, 255)
_C_INSIDE = (0, 255, 0)
_C_IGNORED = (0, 165, 255)   # orange
_C_ERROR = (0, 0, 255)       # red
_C_LABEL_BG = (0, 0, 0)
_TIMELINE_H = 20
_FALLBACK_W, _FALLBACK_H = 1920, 1080


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def detect_trajectory_errors(
    aligned_corridor_path: str,
    raw_json_path: str,
    output_dir: str,
    min_outside_original_frames: int = 3,
    merge_gap_sec: float = 1.0,
    fps: int = 30,
) -> dict[str, Any]:
    """Detect trajectory errors and write outputs to *output_dir*.

    Parameters
    ----------
    aligned_corridor_path:
        Path to ``aligned_corridor.json`` produced by corridor_alignment.py.
    raw_json_path:
        Path to ``raw.json`` for the learner run (provides frame_stride, run_id).
    output_dir:
        Directory where outputs are written (must be the run directory).
    min_outside_original_frames:
        Minimum number of *original* (non-strided) frames an outside window must
        span to be counted as an error.  Shorter windows are treated as noise.
    merge_gap_sec:
        Two outside windows closer than this many seconds of inside time are
        merged into a single error event.
    fps:
        Source video frame rate used for time calculations.

    Returns
    -------
    dict
        The full trajectory_errors payload that was written to disk.
    """
    with open(aligned_corridor_path) as fh:
        aligned: dict[str, Any] = json.load(fh)
    with open(raw_json_path) as fh:
        raw: dict[str, Any] = json.load(fh)

    frame_checks: list[dict[str, Any]] = sorted(
        aligned["frame_checks"], key=lambda c: c["frame_index"]
    )
    frame_stride: int = int(raw["frame_stride"])
    margin_px: float = float(aligned["margin_px"])
    run_id: str = raw["run_id"]
    expert_id: str = aligned["expert_id"]

    # ── Threshold conversion ─────────────────────────────────────────────────
    min_strided = max(1, min_outside_original_frames // frame_stride)
    merge_gap_strided = max(1, int((fps * merge_gap_sec) / frame_stride))

    print("=== TRAJECTORY ERROR DETECTION ===")
    print(f"Frame stride: {frame_stride}")
    print(
        f"Min duration: {min_outside_original_frames} original frames "
        f"({min_strided} strided)"
    )
    print(f"Merge gap: {merge_gap_sec}s ({merge_gap_strided} strided frames)")
    print()

    # ── Step 1: state machine → raw outside windows ──────────────────────────
    raw_windows: list[tuple[int, int]] = []
    in_window = False
    window_start = 0

    for i, check in enumerate(frame_checks):
        if check["outside"] and not in_window:
            in_window = True
            window_start = i
        elif not check["outside"] and in_window:
            in_window = False
            raw_windows.append((window_start, i - 1))

    if in_window:
        raw_windows.append((window_start, len(frame_checks) - 1))

    print(f"Raw outside windows: {len(raw_windows)}")
    for w in raw_windows:
        fi_start = frame_checks[w[0]]["frame_index"]
        fi_end = frame_checks[w[1]]["frame_index"]
        print(f"  frames {fi_start}-{fi_end} ({w[1] - w[0] + 1} strided frames)")

    # ── Step 2: filter by minimum duration ───────────────────────────────────
    filtered_windows: list[tuple[int, int]] = []
    ignored_windows: list[tuple[int, int]] = []

    for start, end in raw_windows:
        if end - start + 1 >= min_strided:
            filtered_windows.append((start, end))
        else:
            ignored_windows.append((start, end))

    print(
        f"\nAfter duration filter: {len(filtered_windows)} kept, "
        f"{len(ignored_windows)} ignored"
    )

    # ── Step 3: merge windows closer than merge_gap_strided ──────────────────
    merged_windows: list[list[int]] = []
    for window in filtered_windows:
        if not merged_windows:
            merged_windows.append(list(window))
        else:
            gap = window[0] - merged_windows[-1][1]
            if gap <= merge_gap_strided:
                merged_windows[-1][1] = window[1]
                print(
                    f"  Merged: gap={gap} strided frames "
                    f"(threshold={merge_gap_strided})"
                )
            else:
                merged_windows.append(list(window))

    print(f"\nAfter merge: {len(merged_windows)} final error events")

    # ── Step 4: build error events ───────────────────────────────────────────
    error_events: list[dict[str, Any]] = []
    print("\n--- Error Events ---")

    for idx, (start, end) in enumerate(merged_windows):
        checks_in_window = frame_checks[start : end + 1]

        frame_start = frame_checks[start]["frame_index"]
        frame_end = frame_checks[end]["frame_index"]
        ts_start = frame_start / fps
        ts_end = frame_end / fps

        peak_check = max(checks_in_window, key=lambda c: c["deviation_px"])
        peak_dev = float(peak_check["deviation_px"])
        mean_dev = float(
            sum(c["deviation_px"] for c in checks_in_window) / len(checks_in_window)
        )
        mean_signed = float(
            sum(c["signed_deviation_px"] for c in checks_in_window)
            / len(checks_in_window)
        )
        direction = "right" if mean_signed > 0 else "left"
        duration_sec = round(ts_end - ts_start, 3)

        error_events.append(
            {
                "error_id": idx + 1,
                "frame_start": frame_start,
                "frame_end": frame_end,
                "timestamp_start_sec": round(ts_start, 3),
                "timestamp_end_sec": round(ts_end, 3),
                "duration_frames": frame_end - frame_start,
                "duration_sec": duration_sec,
                "peak_deviation_px": round(peak_dev, 2),
                "mean_deviation_px": round(mean_dev, 2),
                "direction": direction,
                "peak_location": {
                    "x": round(float(peak_check["learner_x"]), 2),
                    "y": round(float(peak_check["learner_y"]), 2),
                },
                "strided_frames_outside": end - start + 1,
            }
        )
        print(
            f"Error {idx + 1}: frames {frame_start}-{frame_end} | "
            f"{duration_sec}s | peak={round(peak_dev, 1)}px | direction={direction}"
        )

    # ── Step 5: ignored events list ──────────────────────────────────────────
    ignored_events: list[dict[str, Any]] = []
    print("\n--- Ignored (too short) ---")
    for i_idx, (start, end) in enumerate(ignored_windows):
        strided_count = end - start + 1
        fi_start = frame_checks[start]["frame_index"]
        fi_end = frame_checks[end]["frame_index"]
        ignored_events.append(
            {
                "frame_start": fi_start,
                "frame_end": fi_end,
                "reason": "duration_too_short",
                "strided_frames": strided_count,
            }
        )
        print(f"Ignored {i_idx + 1}: frames {fi_start}-{fi_end} | {strided_count} strided frames")

    if not ignored_events:
        print("  (none)")

    # ── Step 6: summary ──────────────────────────────────────────────────────
    total_outside = sum(1 for c in frame_checks if c["outside"])
    total_frames = len(frame_checks)
    mean_duration = (
        sum(e["duration_sec"] for e in error_events) / len(error_events)
        if error_events
        else 0.0
    )
    max_peak = max(
        (e["peak_deviation_px"] for e in error_events), default=0.0
    )

    summary: dict[str, Any] = {
        "total_errors": len(error_events),
        "total_ignored_exits": len(ignored_events),
        "total_outside_frames": total_outside,
        "total_frames_checked": total_frames,
        "outside_percentage": round(100 * total_outside / total_frames, 1)
        if total_frames
        else 0.0,
        "mean_error_duration_sec": round(mean_duration, 3),
        "max_peak_deviation_px": round(max_peak, 2),
    }

    # ── Step 7: save JSON ─────────────────────────────────────────────────────
    output: dict[str, Any] = {
        "run_id": run_id,
        "expert_id": expert_id,
        "error_type": "trajectory",
        "config": {
            "margin_px": margin_px,
            "min_outside_original_frames": min_outside_original_frames,
            "merge_gap_sec": merge_gap_sec,
            "frame_stride": frame_stride,
            "fps": fps,
        },
        "total_errors": len(error_events),
        "error_events": error_events,
        "ignored_events": ignored_events,
        "summary": summary,
    }

    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, TRAJECTORY_ERRORS_FILENAME)
    with open(output_path, "w") as fh:
        json.dump(output, fh, indent=2)
    print(f"\nSaved: {output_path}")

    print(f"\n=== RESULT: {len(error_events)} trajectory errors detected ===")

    # ── Step 8: generate preview image ───────────────────────────────────────
    try:
        preview_path = os.path.join(output_dir, TRAJECTORY_ERRORS_PREVIEW_FILENAME)
        _draw_preview(
            aligned=aligned,
            frame_checks=frame_checks,
            error_events=error_events,
            ignored_windows=ignored_windows,
            merged_windows=merged_windows,
            fps=fps,
            frame_stride=frame_stride,
            output_path=preview_path,
        )
        output["trajectory_errors_preview_path"] = preview_path
        with open(output_path, "w") as fh:
            json.dump(output, fh, indent=2)
    except Exception:  # noqa: BLE001
        logger.exception("[trajectory_errors] Preview generation failed")

    return output


# ─────────────────────────────────────────────────────────────────────────────
# Preview image
# ─────────────────────────────────────────────────────────────────────────────

def _draw_preview(
    *,
    aligned: dict[str, Any],
    frame_checks: list[dict[str, Any]],
    error_events: list[dict[str, Any]],
    ignored_windows: list[tuple[int, int]],
    merged_windows: list[list[int]],
    fps: int,
    frame_stride: int,
    output_path: str,
) -> None:
    """Render the trajectory errors onto a blank canvas and save as PNG."""

    # Determine canvas size from aligned data (prefer video-width metadata)
    width: int = int(aligned.get("video_width", _FALLBACK_W) or _FALLBACK_W)
    height: int = int(aligned.get("video_height", _FALLBACK_H) or _FALLBACK_H)
    if width <= 0:
        width = _FALLBACK_W
    if height <= 0:
        height = _FALLBACK_H

    canvas = np.zeros((height + _TIMELINE_H, width, 3), dtype=np.uint8)

    # ── Corridor polygon fill ─────────────────────────────────────────────────
    corridor_polygon: list[list[float]] = aligned.get("corridor_polygon", [])
    if corridor_polygon:
        pts = np.array(
            [[int(p["x"]), int(p["y"])] if isinstance(p, dict) else [int(p[0]), int(p[1])]
             for p in corridor_polygon],
            dtype=np.int32,
        )
        overlay = canvas[:height].copy()
        cv2.fillPoly(overlay, [pts], _C_CORRIDOR_FILL)
        cv2.addWeighted(overlay, 0.3, canvas[:height], 0.7, 0, canvas[:height])

    # ── Corridor centerline ───────────────────────────────────────────────────
    centerline: list[dict[str, Any]] = aligned.get("aligned_centerline", [])
    if len(centerline) >= 2:
        cl_pts = np.array(
            [[int(p["x"]), int(p["y"])] for p in centerline], dtype=np.int32
        )
        cv2.polylines(canvas[:height], [cl_pts], False, _C_CENTERLINE, 2, cv2.LINE_AA)

    # ── Build set of frame indices per category ───────────────────────────────
    # Ignored outside indices (strided list positions)
    ignored_strided_idx: set[int] = set()
    for start, end in ignored_windows:
        ignored_strided_idx.update(range(start, end + 1))

    # Error (merged window) strided indices
    error_strided_idx: set[int] = set()
    for start, end in merged_windows:
        error_strided_idx.update(range(start, end + 1))

    # ── Draw per-frame dots ───────────────────────────────────────────────────
    for i, check in enumerate(frame_checks):
        cx = int(check["learner_x"])
        cy = int(check["learner_y"])
        if cy >= height:
            continue
        if i in error_strided_idx:
            cv2.circle(canvas, (cx, cy), 6, _C_ERROR, -1, cv2.LINE_AA)
        elif i in ignored_strided_idx:
            cv2.circle(canvas, (cx, cy), 4, _C_IGNORED, -1, cv2.LINE_AA)
        else:
            cv2.circle(canvas, (cx, cy), 3, _C_INSIDE, -1, cv2.LINE_AA)

    # ── Error labels at peak location ─────────────────────────────────────────
    for ev in error_events:
        px = int(ev["peak_location"]["x"])
        py = int(ev["peak_location"]["y"])
        if py >= height:
            py = height - 20
        label = f"ERROR {ev['error_id']}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale, thickness = 0.7, 2
        (tw, th), baseline = cv2.getTextSize(label, font, scale, thickness)
        # white outline
        cv2.putText(canvas, label, (px - tw // 2, py - 12),
                    font, scale, (255, 255, 255), thickness + 2, cv2.LINE_AA)
        # red text
        cv2.putText(canvas, label, (px - tw // 2, py - 12),
                    font, scale, _C_ERROR, thickness, cv2.LINE_AA)

    # ── Timeline bar ─────────────────────────────────────────────────────────
    if frame_checks:
        last_frame = frame_checks[-1]["frame_index"]
        total_original_frames = max(last_frame + frame_stride, 1)

        # Map from strided index → x pixel range on timeline
        def _fi_to_x(fi: int) -> int:
            return int(fi / total_original_frames * width)

        # Category lookup for each frame check by frame_index
        error_frame_set: set[int] = set()
        for ev in error_events:
            for fi in range(ev["frame_start"], ev["frame_end"] + frame_stride, frame_stride):
                error_frame_set.add(fi)

        ignored_frame_set: set[int] = set()
        for start, end in ignored_windows:
            for idx in range(start, end + 1):
                ignored_frame_set.add(frame_checks[idx]["frame_index"])

        tl_y = height  # top of timeline strip

        # Draw each frame check as a 1px-wide column
        for check in frame_checks:
            fi = check["frame_index"]
            x0 = _fi_to_x(fi)
            x1 = _fi_to_x(fi + frame_stride)
            x1 = max(x1, x0 + 1)
            x1 = min(x1, width)

            if fi in error_frame_set:
                color = _C_ERROR
            elif fi in ignored_frame_set:
                color = _C_IGNORED
            else:
                color = _C_INSIDE

            canvas[tl_y : tl_y + _TIMELINE_H, x0:x1] = color

        # White vertical tick marks every 1 second
        if fps > 0:
            tick_interval_frames = fps
            fi = 0
            while fi <= last_frame:
                x = _fi_to_x(fi)
                cv2.line(canvas, (x, tl_y), (x, tl_y + _TIMELINE_H - 1),
                         (255, 255, 255), 1)
                fi += tick_interval_frames

        # "E1", "E2" labels above each red segment
        for ev in error_events:
            x_mid = (_fi_to_x(ev["frame_start"]) + _fi_to_x(ev["frame_end"])) // 2
            label = f"E{ev['error_id']}"
            font = cv2.FONT_HERSHEY_SIMPLEX
            scale, thickness = 0.4, 1
            (tw, _th), _ = cv2.getTextSize(label, font, scale, thickness)
            tx = max(0, x_mid - tw // 2)
            ty = tl_y - 4
            cv2.putText(canvas, label, (tx, ty),
                        font, scale, _C_ERROR, thickness, cv2.LINE_AA)

    cv2.imwrite(output_path, canvas)
    print(f"Saved preview: {output_path}")
