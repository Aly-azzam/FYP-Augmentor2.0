"""Convert vibration detection results into unified-error-compatible dicts.

Reads the summary dict returned by ``run_vibration_detection()`` and produces
one error dict per vibration event, using the same field schema written by
``merge_errors.merge_errors()`` into ``unified_errors.json``.

Each vibration event maps 1-to-1 to one error — events are never merged here.
Bounding boxes are computed from YOLO detections that fall within the event's
frame range, so the click target covers the region where scissors appeared
during the vibration period.

Typical call site (evaluation orchestrator):

    from app.services.evaluation.vibration_error_detector import detect_vibration_errors

    vibration_errors = detect_vibration_errors(
        vibration_summary=vibration_summary,
        yolo_roi_data=yolo_result["all_detections"],
        fps=fps,
        frame_width=frame_width,
        frame_height=frame_height,
    )
    # Then inject vibration_errors into the unified error list before saving.
"""
from __future__ import annotations

from typing import Any


# Padding added around the raw YOLO bbox union.
_BBOX_PADDING = 150


# ── Bounding box helpers ──────────────────────────────────────────────────────

def _compute_vibration_bbox(
    frame_start: int,
    frame_end: int,
    yolo_roi_data: list[dict],
    frame_width: int,
    frame_height: int,
    padding: int = _BBOX_PADDING,
) -> dict[str, float]:
    """Compute a click-target bounding box for one vibration event.

    Takes the union of all raw YOLO scissors bboxes (the ``bbox`` field from
    yolo_roi_data, i.e. [x1, y1, x2, y2] as detected) for every frame in
    [frame_start, frame_end] where detected=True.  Adds *padding* px on each
    side (matching roi_padding_px=40 in the standalone optical flow config)
    and clamps to frame boundaries.

    Example from the standalone log:
        bbox=[864.75, 274.5, 968.25, 858.0]
        → with 40 px padding: (824.75, 234.5) – (1008.25, 898.0)

    Fallback when no detections exist in the range: centre quarter of the frame.
    """
    xs1: list[float] = []
    ys1: list[float] = []
    xs2: list[float] = []
    ys2: list[float] = []

    for det in yolo_roi_data:
        fi = int(det.get("frame_index", -1))
        if fi < frame_start or fi > frame_end:
            continue
        if not det.get("detected", False):
            continue
        bbox = det.get("bbox")
        if not bbox or len(bbox) != 4:
            continue
        x1, y1, x2, y2 = (float(v) for v in bbox)
        if x2 <= x1 or y2 <= y1:
            continue
        xs1.append(x1)
        ys1.append(y1)
        xs2.append(x2)
        ys2.append(y2)

    print(f"[BBOX DEBUG] event frames {frame_start}–{frame_end}, matched {len(xs1)} YOLO detections", flush=True)

    if xs1:
        raw_x_min = min(xs1) - padding
        raw_y_min = min(ys1) - padding
        raw_x_max = max(xs2) + padding
        raw_y_max = max(ys2) + padding
    else:
        raw_x_min = frame_width  * 0.20
        raw_y_min = frame_height * 0.20
        raw_x_max = frame_width  * 0.80
        raw_y_max = frame_height * 0.80

    return {
        "x_min": round(max(0.0,               raw_x_min), 2),
        "y_min": round(max(0.0,               raw_y_min), 2),
        "x_max": round(min(float(frame_width),  raw_x_max), 2),
        "y_max": round(min(float(frame_height), raw_y_max), 2),
    }


def _bbox_center(bbox: dict[str, float]) -> dict[str, float]:
    """Return {"x": cx, "y": cy} for the centre of a bounding box dict."""
    return {
        "x": round((bbox["x_min"] + bbox["x_max"]) / 2.0, 2),
        "y": round((bbox["y_min"] + bbox["y_max"]) / 2.0, 2),
    }


# ── Public API ────────────────────────────────────────────────────────────────

def detect_vibration_errors(
    vibration_summary: dict[str, Any],
    yolo_roi_data: list[dict],
    fps: float,
    frame_width: int,
    frame_height: int,
) -> list[dict[str, Any]]:
    """Convert vibration events into unified-error-compatible dicts.

    One vibration event → one error dict.  Events are never merged with each
    other or with trajectory/angle errors here.  The caller (typically the
    evaluation orchestrator) is responsible for assigning ``error_id`` values
    before writing ``unified_errors.json``.

    Parameters
    ----------
    vibration_summary:
        The dict returned by ``run_vibration_detection()``.  Must contain the
        keys ``vibration_detected`` (bool) and ``vibration_events`` (list).
    yolo_roi_data:
        The ``all_detections`` list from ``run_shared_yolo()`` — same list
        passed to ``run_vibration_detection()``.  Used to build click-target
        bounding boxes from real scissors positions.
    fps:
        Video frames per second.  Used only if timestamps need to be derived
        from frame indices (normally they are already in the event dict).
    frame_width:
        Video frame width in pixels — for clamping and fallback bbox.
    frame_height:
        Video frame height in pixels — for clamping and fallback bbox.

    Returns
    -------
    list[dict]
        One entry per vibration event.  Returns an empty list when
        ``vibration_summary["vibration_detected"]`` is False or when there
        are no events.

        Each dict matches the schema written by merge_errors.merge_errors()
        into ``unified_errors.json``:

        {
            "error_id":            None,          # assigned by caller
            "error_type":          "vibration",
            "frame_start":         int,
            "frame_end":           int,
            "timestamp_start_sec": float,
            "timestamp_end_sec":   float,
            "duration_sec":        float,
            "peak_location":       {"x": float, "y": float},
            "bounding_box":        {"x_min": float, "y_min": float,
                                    "x_max": float, "y_max": float},
            "area_radius_px":      None,
            "direction":           None,
            "peak_deviation_px":   None,
            "peak_angle_diff_deg": None,
            "mean_angle_diff_deg": None,
            # vibration-specific extras (ignored by the scoring engine)
            "dominant_freq_hz":    float,
            "peak_confidence":     float,
            "severity":            str,   # "mild" | "moderate" | "severe"
            "consecutive_windows": int,
        }
    """
    # Early exit when nothing was detected
    if not vibration_summary.get("vibration_detected", False):
        return []

    events: list[dict] = vibration_summary.get("vibration_events", [])
    if not events:
        return []

    errors: list[dict[str, Any]] = []

    for event in events:
        frame_start = int(event["start_frame"])
        frame_end   = int(event["end_frame"])

        # Timestamps — read from event; derive from frames if missing
        ts_start = float(
            event.get("timestamp_start_sec")
            if event.get("timestamp_start_sec") is not None
            else frame_start / max(fps, 1e-6)
        )
        ts_end = float(
            event.get("timestamp_end_sec")
            if event.get("timestamp_end_sec") is not None
            else frame_end / max(fps, 1e-6)
        )
        duration = round(
            float(event.get("duration_sec") if event.get("duration_sec") is not None
                  else ts_end - ts_start),
            3,
        )

        bbox = _compute_vibration_bbox(
            frame_start=frame_start,
            frame_end=frame_end,
            yolo_roi_data=yolo_roi_data,
            frame_width=frame_width,
            frame_height=frame_height,
        )
        peak_location = _bbox_center(bbox)

        errors.append({
            # ── Core fields — match merge_errors.py schema exactly ────────────
            "error_id":            None,          # assigned by caller before saving
            "error_type":          "vibration",
            "frame_start":         frame_start,
            "frame_end":           frame_end,
            "timestamp_start_sec": round(ts_start, 3),
            "timestamp_end_sec":   round(ts_end, 3),
            "duration_sec":        duration,
            "peak_location":       peak_location,
            "bounding_box":        bbox,
            "area_radius_px":      None,
            "direction":           None,
            "peak_deviation_px":   None,
            "peak_angle_diff_deg": None,
            "mean_angle_diff_deg": None,
            # ── Vibration-specific extras (transparent to scoring engine) ─────
            "dominant_freq_hz":    float(event.get("dominant_freq_hz", 0.0)),
            "peak_confidence":     float(event.get("peak_confidence", 0.0)),
            "severity":            str(event.get("severity", "mild")),
            "consecutive_windows": int(event.get("consecutive_windows", 0)),
        })

    return errors
