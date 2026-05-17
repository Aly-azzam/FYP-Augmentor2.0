"""Angle error detection engine for AugMentor 2.0.

Reads DTW alignment and corridor data, applies state-machine error detection,
and writes angle_errors.json to the evaluation output directory.

Inputs
------
storage/evaluation/{run_id}/angle/dtw_alignment.json
storage/evaluation/{run_id}/trajectory/aligned_corridor.json  (optional)

Output
------
storage/evaluation/{run_id}/angle/angle_errors.json
"""

from __future__ import annotations

import json
import os
from pathlib import Path


# ─────────────────────────────────────────────────────────────────────────────
# Frame-index helpers
# ─────────────────────────────────────────────────────────────────────────────

def nearest_strided_frame(frame_idx: int, stride: int = 5) -> int:
    """Map a DTW frame index to the nearest strided corridor frame index.

    The corridor uses strided frame indices (multiples of *stride*, e.g.
    0, 5, 10 …) while DTW uses original frame indices (0, 1, 2, 3 …).
    """
    return round(frame_idx / stride) * stride


# ─────────────────────────────────────────────────────────────────────────────
# Main detection function
# ─────────────────────────────────────────────────────────────────────────────

def detect_angle_errors(
    dtw_alignment_path: str,
    aligned_corridor_path: str,
    output_dir: str,
    angle_threshold_deg: float = 15.0,
    min_duration_frames: int = 10,
    merge_gap_sec: float = 0.5,
    fps: int = 30,
) -> dict:
    """Detect sustained angle errors inside the expert corridor.

    Rules
    -----
    1. Corridor filter: only frames where ``inside_corridor=True`` are
       considered; outside frames are trajectory errors, not angle errors.
    2. Angle threshold: ``angle_difference > angle_threshold_deg`` (default 15°).
    3. Minimum duration: must exceed the threshold for at least
       *min_duration_frames* consecutive frames (default 10).
    4. Look-ahead decision on re-entry: when the learner re-enters the corridor
       after a trajectory error, the next ``LOOK_AHEAD_FRAMES`` (75) frames are
       buffered.  If the angle drops below the threshold and stays there for at
       least ``MIN_CORRECTION_FRAMES`` (10) consecutive frames within the buffer
       the correction is accepted and the buffer is discarded (no error).
       Otherwise the buffer is processed normally by the state machine.
    5. Merge rule: two error windows separated by ≤ *merge_gap_sec* are merged.
    6. State machine: ``OK`` → ``ERROR`` → ``OK``.

    Parameters
    ----------
    dtw_alignment_path:
        Path to ``dtw_alignment.json``.
    aligned_corridor_path:
        Path to ``aligned_corridor.json`` (may be absent; errors are still
        detected but ``inside_corridor`` will be ``None`` for every frame).
    output_dir:
        Directory where ``angle_errors.json`` is written.
    angle_threshold_deg:
        Minimum absolute angle difference to be considered an error candidate.
    min_duration_frames:
        Minimum consecutive frames above threshold to open an error event.
    merge_gap_sec:
        Maximum gap (seconds) between two error windows for them to be merged.
    fps:
        Frames per second — used for display only (timestamps are from DTW).

    Returns
    -------
    dict
        The full output payload that was also written to ``angle_errors.json``.
    """

    # ── Load inputs ───────────────────────────────────────────────────────
    with open(dtw_alignment_path) as fh:
        dtw = json.load(fh)
    matches: list[dict] = dtw["matches"]

    corridor_available = Path(aligned_corridor_path).exists()
    corridor_lookup: dict[int, dict] = {}

    if corridor_available:
        with open(aligned_corridor_path) as fh:
            corridor = json.load(fh)
        corridor_lookup = {
            fc["frame_index"]: fc
            for fc in corridor["frame_checks"]
        }

    # ── Header ────────────────────────────────────────────────────────────
    print("=== ANGLE ERROR DETECTION ===")
    print(f"Total DTW matches: {len(matches)}")

    # ── Enrich each match with corridor data + just_re_entered flag ───────
    # Initialize based on actual first frame status, not assumed outside
    first_match = matches[0] if matches else None
    if first_match and corridor_available:
        nearest = nearest_strided_frame(first_match["learner_frame_index"])
        sam_info = corridor_lookup.get(nearest, {})
        prev_outside = sam_info.get("outside", True)
    else:
        prev_outside = True
    for match in matches:
        if corridor_available:
            nearest = nearest_strided_frame(match["learner_frame_index"])
            sam_info = corridor_lookup.get(nearest, {})
            match["inside_corridor"] = not sam_info.get("outside", True)
            match["learner_x"] = sam_info.get("learner_x", None)
            match["learner_y"] = sam_info.get("learner_y", None)
            match["distance_from_corridor_edge"] = sam_info.get("deviation_px", None)
        else:
            match["inside_corridor"] = None
            match["learner_x"] = None
            match["learner_y"] = None
            match["distance_from_corridor_edge"] = None

        current_outside = (
            not match["inside_corridor"]
            if match["inside_corridor"] is not None
            else True
        )
        match["just_re_entered"] = prev_outside is True and current_outside is False
        prev_outside = current_outside

    total_inside = sum(1 for m in matches if m.get("inside_corridor"))
    print(f"Inside corridor matches: {total_inside}")
    print(f"Corridor data: {'available' if corridor_available else 'not available'}")
    print()

    # ── State-machine with look-ahead re-entry decision ───────────────────
    LOOK_AHEAD_FRAMES = 180
    MIN_CORRECTION_FRAMES = 10  # consecutive frames below threshold = corrected

    raw_windows: list[tuple[int, int]] = []
    in_window = False
    window_start: int | None = None

    i = 0
    while i < len(matches):
        match = matches[i]

        # Rule 1 — skip frames outside corridor
        if not match.get("inside_corridor"):
            if in_window:
                in_window = False
                raw_windows.append((window_start, i - 1))  # type: ignore[arg-type]
                window_start = None
            i += 1
            continue

        # Rule 4 — look-ahead decision on re-entry
        if match.get("just_re_entered"):
            # Close any window that was open before the trajectory error
            if in_window:
                in_window = False
                raw_windows.append((window_start, i - 1))  # type: ignore[arg-type]
                window_start = None

            # Collect look-ahead buffer (up to LOOK_AHEAD_FRAMES inside corridor)
            buffer: list[tuple[int, dict]] = []
            j = i
            while j < len(matches) and j < i + LOOK_AHEAD_FRAMES:
                m = matches[j]
                if not m.get("inside_corridor"):
                    break   # buffer stops if learner exits again
                buffer.append((j, m))
                j += 1

            # Decide: did the learner correct their angle within the buffer?
            corrected = False
            consecutive_ok = 0
            for _, m in buffer:
                if m["angle_difference"] <= angle_threshold_deg:
                    consecutive_ok += 1
                    if consecutive_ok >= MIN_CORRECTION_FRAMES:
                        corrected = True
                        break
                else:
                    consecutive_ok = 0

            print(f"[DEBUG just_re_entered] frame_idx={match['learner_frame_index']} buffer_size={len(buffer)} corrected={corrected}")
            if corrected:
                # Correction accepted — skip the entire buffer, no error opened
                i = j
                continue
            # Correction failed — fall through and process buffer frames normally

        # Rule 2 / Rule 6 — normal threshold state machine
        is_error = match["angle_difference"] > angle_threshold_deg

        if is_error and not in_window:
            in_window = True
            window_start = i
        elif not is_error and in_window:
            in_window = False
            raw_windows.append((window_start, i - 1))  # type: ignore[arg-type]
            window_start = None

        i += 1

    # Close any window still open at the end
    if in_window and window_start is not None:
        raw_windows.append((window_start, len(matches) - 1))

    print(f"Raw angle error windows: {len(raw_windows)}")

    # ── Rule 3 — minimum duration filter ──────────────────────────────────
    filtered_windows: list[tuple[int, int]] = []
    ignored_windows: list[tuple[int, int]] = []

    for start, end in raw_windows:
        duration = end - start + 1
        if duration >= min_duration_frames:
            filtered_windows.append((start, end))
        else:
            ignored_windows.append((start, end))

    print(
        f"After duration filter: {len(filtered_windows)} kept, "
        f"{len(ignored_windows)} ignored"
    )

    # ── Rule 5 — merge close windows ─────────────────────────────────────
    merged_windows: list[list[int]] = []

    for window in filtered_windows:
        if not merged_windows:
            merged_windows.append(list(window))
        else:
            gap_sec = (
                matches[window[0]]["learner_timestamp_sec"]
                - matches[merged_windows[-1][1]]["learner_timestamp_sec"]
            )
            if gap_sec <= merge_gap_sec:
                merged_windows[-1][1] = window[1]
                print(f"  Merged: gap={round(gap_sec, 2)}s")
            else:
                merged_windows.append(list(window))

    print(f"After merge: {len(merged_windows)} final angle errors")

    # ── Build error events ────────────────────────────────────────────────
    print()
    print("--- Angle Error Events ---")

    error_events: list[dict] = []

    for idx, (start, end) in enumerate(merged_windows):
        window_matches = matches[start: end + 1]

        peak_match = max(window_matches, key=lambda m: m["angle_difference"])
        peak_diff = peak_match["angle_difference"]
        mean_diff = (
            sum(m["angle_difference"] for m in window_matches) / len(window_matches)
        )

        frame_start = window_matches[0]["learner_frame_index"]
        frame_end = window_matches[-1]["learner_frame_index"]
        ts_start = window_matches[0]["learner_timestamp_sec"]
        ts_end = window_matches[-1]["learner_timestamp_sec"]

        # Centroid of all valid positions in the window — used as circle center
        valid_positions = [
            (m.get("learner_x"), m.get("learner_y"))
            for m in window_matches
            if m.get("learner_x") is not None and m.get("learner_y") is not None
        ]
        if valid_positions:
            center_x = sum(p[0] for p in valid_positions) / len(valid_positions)
            center_y = sum(p[1] for p in valid_positions) / len(valid_positions)
        else:
            center_x = peak_match.get("learner_x")
            center_y = peak_match.get("learner_y")

        event = {
            "error_id": idx + 1,
            "error_type": "angle",
            "frame_start": frame_start,
            "frame_end": frame_end,
            "timestamp_start_sec": round(ts_start, 3),
            "timestamp_end_sec": round(ts_end, 3),
            "duration_frames": frame_end - frame_start,
            "duration_sec": round(ts_end - ts_start, 3),
            "peak_angle_diff_deg": round(peak_diff, 2),
            "mean_angle_diff_deg": round(mean_diff, 2),
            "peak_frame": peak_match["learner_frame_index"],
            "peak_location": {
                "x": round(center_x, 2) if center_x is not None else None,
                "y": round(center_y, 2) if center_y is not None else None,
            },
            "area_radius_px": 80,
            "frames_in_window": end - start + 1,
        }
        error_events.append(event)

        print(
            f"Angle Error {idx + 1}: "
            f"frames {frame_start}-{frame_end} | "
            f"{round(ts_end - ts_start, 2)}s | "
            f"peak={round(peak_diff, 1)}° | "
            f"mean={round(mean_diff, 1)}°"
        )

    # ── Ignored events list ───────────────────────────────────────────────
    ignored_events: list[dict] = [
        {
            "frame_start": matches[s]["learner_frame_index"],
            "frame_end": matches[e]["learner_frame_index"],
            "reason": "duration_too_short",
            "frames": e - s + 1,
        }
        for s, e in ignored_windows
    ]

    # ── Summary ───────────────────────────────────────────────────────────
    summary = {
        "total_errors": len(error_events),
        "total_ignored": len(ignored_events),
        "total_matches": len(matches),
        "total_inside_corridor_matches": total_inside,
        "corridor_data_available": corridor_available,
        "mean_error_duration_sec": (
            round(
                sum(e["duration_sec"] for e in error_events) / len(error_events),
                3,
            )
            if error_events
            else 0
        ),
        "max_peak_angle_diff_deg": (
            round(max(e["peak_angle_diff_deg"] for e in error_events), 2)
            if error_events
            else 0
        ),
    }

    # ── Assemble full output ──────────────────────────────────────────────
    output = {
        "run_id": str(Path(output_dir).parent.name),
        "error_type": "angle",
        "config": {
            "angle_threshold_deg": angle_threshold_deg,
            "min_duration_frames": min_duration_frames,
            "merge_gap_sec": merge_gap_sec,
            "look_ahead_frames": LOOK_AHEAD_FRAMES,
            "min_correction_frames": MIN_CORRECTION_FRAMES,
            "fps": fps,
        },
        "total_errors": len(error_events),
        "error_events": error_events,
        "ignored_events": ignored_events,
        "summary": summary,
    }

    # ── Write output ──────────────────────────────────────────────────────
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "angle_errors.json")
    with open(output_path, "w") as fh:
        json.dump(output, fh, indent=2)

    # ── Preview image ─────────────────────────────────────────────────────
    preview_path = os.path.join(output_dir, "angle_errors_preview.png")
    _generate_angle_errors_preview(
        preview_path=preview_path,
        aligned_corridor_path=aligned_corridor_path,
        matches=matches,
        error_events=error_events,
        fps=fps,
    )

    print()
    print(f"=== RESULT: {len(error_events)} angle errors detected ===")
    print(f"Saved: {output_path}")
    print(f"Preview: {preview_path}")

    return output


# ─────────────────────────────────────────────────────────────────────────────
# Preview image generator
# ─────────────────────────────────────────────────────────────────────────────

def _generate_angle_errors_preview(
    preview_path: str,
    aligned_corridor_path: str,
    matches: list[dict],
    error_events: list[dict],
    fps: int = 30,
) -> None:
    """Render a 1920 × 1080 summary image for the detected angle errors.

    Layers (bottom → top)
    ---------------------
    1. Semi-transparent green filled corridor polygon
    2. White aligned centerline
    3. Grey dots — DTW matches outside corridor
    4. Green dots — DTW matches inside corridor
    5. Red dots   — DTW matches inside an error window
    6. Red circles + numbered labels at each error peak location
    7. Timeline bar (bottom 20 px): green / grey / red segments + 1-second ticks
    """
    try:
        import cv2  # type: ignore[import]
        import numpy as np
    except ImportError:
        print("[angle_errors] WARNING: opencv-python not available — skipping preview")
        return

    W, H = 1920, 1080
    TIMELINE_H = 20
    TIMELINE_Y = H - TIMELINE_H

    canvas = np.zeros((H, W, 3), dtype=np.uint8)

    # ── 1. Corridor polygon & centerline ─────────────────────────────────
    corridor_available = Path(aligned_corridor_path).exists()
    if corridor_available:
        try:
            with open(aligned_corridor_path) as fh:
                corridor_data = json.load(fh)

            # Semi-transparent green fill
            poly_raw = corridor_data.get("corridor_polygon", [])
            if len(poly_raw) >= 3:
                poly_arr = np.array(
                    [[int(p["x"]), int(p["y"])] for p in poly_raw], dtype=np.int32
                )
                overlay = canvas.copy()
                cv2.fillPoly(overlay, [poly_arr], (0, 180, 0))
                cv2.addWeighted(overlay, 0.3, canvas, 0.7, 0, dst=canvas)

            # White centerline
            cl_raw = corridor_data.get("aligned_centerline", [])
            cl_pts = [(int(p["x"]), int(p["y"])) for p in cl_raw]
            for i in range(len(cl_pts) - 1):
                cv2.line(canvas, cl_pts[i], cl_pts[i + 1], (255, 255, 255), 2, cv2.LINE_AA)
        except Exception as exc:
            print(f"[angle_errors] WARNING: could not draw corridor: {exc}")

    # ── Build error-frame lookup ──────────────────────────────────────────
    error_frame_set: set[int] = set()
    for ev in error_events:
        for fi in range(ev["frame_start"], ev["frame_end"] + 1):
            error_frame_set.add(fi)

    # ── 2–4. DTW match dots ───────────────────────────────────────────────
    # Draw in order: outside (grey) → inside (green) → error (red)
    # so error dots always appear on top.
    outside_pts: list[tuple[int, int]] = []
    inside_pts: list[tuple[int, int]] = []
    error_pts: list[tuple[int, int]] = []

    for m in matches:
        lx = m.get("learner_x")
        ly = m.get("learner_y")
        if lx is None or ly is None:
            continue
        pt = (int(round(lx)), int(round(ly)))
        fi = m.get("learner_frame_index", -1)
        if fi in error_frame_set:
            error_pts.append(pt)
        elif m.get("inside_corridor"):
            inside_pts.append(pt)
        else:
            outside_pts.append(pt)

    for pt in outside_pts:
        cv2.circle(canvas, pt, 3, (100, 100, 100), -1)   # grey
    for pt in inside_pts:
        cv2.circle(canvas, pt, 3, (0, 200, 60), -1)       # green
    for pt in error_pts:
        cv2.circle(canvas, pt, 4, (0, 50, 220), -1)       # red (BGR)

    # ── 5. Error event markers ────────────────────────────────────────────
    for ev in error_events:
        px = ev.get("peak_location", {}).get("x")
        py = ev.get("peak_location", {}).get("y")
        if px is None or py is None:
            continue
        cx, cy = int(round(px)), int(round(py))
        radius = ev.get("area_radius_px", 80)
        error_id = ev.get("error_id", "?")
        label = f"ANGLE ERROR {error_id}"

        # Filled red circle outline
        cv2.circle(canvas, (cx, cy), radius, (0, 0, 220), 2, cv2.LINE_AA)

        # Text: white outline then red fill
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 0.75
        text_x = cx - radius
        text_y = cy - radius - 8
        cv2.putText(canvas, label, (text_x, text_y), font, scale, (255, 255, 255), 4, cv2.LINE_AA)
        cv2.putText(canvas, label, (text_x, text_y), font, scale, (0, 0, 220),   2, cv2.LINE_AA)

    # ── 6. Timeline bar ───────────────────────────────────────────────────
    if matches:
        max_ts = max(m.get("learner_timestamp_sec", 0.0) for m in matches)
        max_ts = max_ts if max_ts > 0 else 1.0

        # Background
        cv2.rectangle(canvas, (0, TIMELINE_Y), (W, H), (30, 30, 30), -1)

        for m in matches:
            ts = m.get("learner_timestamp_sec", 0.0)
            fi = m.get("learner_frame_index", -1)
            x = int(ts / max_ts * (W - 1))
            if fi in error_frame_set:
                col = (0, 50, 220)      # red
            elif m.get("inside_corridor"):
                col = (0, 150, 50)      # green
            else:
                col = (90, 90, 90)      # grey
            cv2.line(canvas, (x, TIMELINE_Y), (x, H - 1), col, 1)

        # White tick marks every 1 second
        t_sec = 1.0
        while t_sec <= max_ts:
            tx = int(t_sec / max_ts * (W - 1))
            cv2.line(canvas, (tx, TIMELINE_Y), (tx, H - 1), (255, 255, 255), 1)
            cv2.putText(
                canvas,
                f"{int(t_sec)}s",
                (tx + 2, H - 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.35,
                (200, 200, 200),
                1,
                cv2.LINE_AA,
            )
            t_sec += 1.0

    # ── Save ──────────────────────────────────────────────────────────────
    cv2.imwrite(preview_path, canvas)
    print(f"[angle_errors] Preview saved: {preview_path}")
