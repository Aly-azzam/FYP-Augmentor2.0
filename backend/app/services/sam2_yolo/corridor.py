"""Expert corridor generation from a smoothed YOLO+SAM2 trajectory.

Workflow
--------
1. Load trajectory_smoothed.json (uses smoothed_x / smoothed_y).
2. Compute per-point tangent using central difference (forward/backward at ends).
3. Offset each point left and right by *margin_px* along the normal.
4. Build a closed polygon and save corridor.json.
5. Render a corridor_preview.png (static) and corridor_overlay.mp4 (per-frame).

This module is **expert-only and offline** – it must NOT be imported from any
learner upload or evaluation path.
"""

from __future__ import annotations

import json
import logging
import math
from pathlib import Path
from typing import Any

import cv2
import numpy as np

logger = logging.getLogger(__name__)

CORRIDOR_FILENAME = "corridor.json"
CORRIDOR_PREVIEW_FILENAME = "corridor_preview.png"
CORRIDOR_OVERLAY_FILENAME = "corridor_overlay.mp4"

_POLY_ALPHA = 0.28          # semi-transparency for filled polygon
_COLOR_CORRIDOR = (0, 200, 80)   # green fill
_COLOR_CENTERLINE = (255, 255, 0)  # yellow
_COLOR_LEFT = (0, 140, 255)       # orange-blue
_COLOR_RIGHT = (0, 70, 200)       # darker blue
_COLOR_START = (0, 255, 0)        # bright green dot
_COLOR_END = (0, 0, 255)          # red dot


# ──────────────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────────────

def build_expert_corridor(
    smoothed_trajectory_path: str,
    output_dir: str,
    margin_px: int = 40,
) -> dict[str, Any]:
    """Build expert corridor from trajectory_smoothed.json.

    Parameters
    ----------
    smoothed_trajectory_path:
        Absolute or relative path to trajectory_smoothed.json.
    output_dir:
        Directory where corridor.json, corridor_preview.png and
        corridor_overlay.mp4 are written.
    margin_px:
        Half-width of corridor in pixels (offset left and right from
        centerline).

    Returns
    -------
    The corridor JSON payload (dict).
    """
    smoothed_path = Path(smoothed_trajectory_path).expanduser().resolve()
    out_dir = Path(output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    payload = json.loads(smoothed_path.read_text(encoding="utf-8"))
    expert_code = payload.get("run_id", out_dir.name)
    points_raw = payload.get("points", [])
    if len(points_raw) < 3:
        raise ValueError(
            f"Need at least 3 smoothed points to build a corridor; "
            f"got {len(points_raw)}"
        )

    xs = np.array([p["smoothed_x"] for p in points_raw], dtype=float)
    ys = np.array([p["smoothed_y"] for p in points_raw], dtype=float)
    n = len(xs)

    # ── 1. Tangents ──────────────────────────────────────────────────────────
    tx = np.empty(n, dtype=float)
    ty = np.empty(n, dtype=float)
    tx[0], ty[0] = xs[1] - xs[0], ys[1] - ys[0]
    tx[-1], ty[-1] = xs[-1] - xs[-2], ys[-1] - ys[-2]
    tx[1:-1] = xs[2:] - xs[:-2]
    ty[1:-1] = ys[2:] - ys[:-2]

    # ── 2. Normalize ─────────────────────────────────────────────────────────
    lengths = np.hypot(tx, ty)
    zero_mask = lengths < 1e-9
    warnings: list[str] = []
    if zero_mask.any():
        bad = int(zero_mask.sum())
        warnings.append(
            f"{bad} zero-length tangent(s) at point indices "
            f"{np.where(zero_mask)[0].tolist()[:10]}; tangent replaced with (1,0)."
        )
        tx[zero_mask] = 1.0
        ty[zero_mask] = 0.0
        lengths[zero_mask] = 1.0

    tx /= lengths
    ty /= lengths

    # ── 3. Normals ───────────────────────────────────────────────────────────
    nx = -ty
    ny = tx

    # ── 4. Edges ─────────────────────────────────────────────────────────────
    lx = xs + margin_px * nx
    ly = ys + margin_px * ny
    rx = xs - margin_px * nx
    ry = ys - margin_px * ny

    # ── 5. Closed polygon ────────────────────────────────────────────────────
    poly_x = np.concatenate([lx, rx[::-1], [lx[0]]])
    poly_y = np.concatenate([ly, ry[::-1], [ly[0]]])

    # ── 6. Validation ────────────────────────────────────────────────────────
    all_finite = bool(
        np.isfinite(lx).all()
        and np.isfinite(ly).all()
        and np.isfinite(rx).all()
        and np.isfinite(ry).all()
    )
    if not all_finite:
        warnings.append("Non-finite values found in corridor edges.")

    polygon_closed = bool(
        math.isclose(poly_x[0], poly_x[-1], abs_tol=1e-6)
        and math.isclose(poly_y[0], poly_y[-1], abs_tol=1e-6)
    )

    polygon_pts = np.column_stack([poly_x[:-1], poly_y[:-1]])
    corridor_area = float(abs(_polygon_area(polygon_pts)))
    if corridor_area < 100.0:
        warnings.append(f"Corridor area is very small: {corridor_area:.1f} px².")

    if n < 20:
        warnings.append(f"Centerline has only {n} points (<20); corridor may be unreliable.")

    all_xs = np.concatenate([lx, rx])
    all_ys = np.concatenate([ly, ry])
    bbox = {
        "x_min": round(float(all_xs.min()), 3),
        "x_max": round(float(all_xs.max()), 3),
        "y_min": round(float(all_ys.min()), 3),
        "y_max": round(float(all_ys.max()), 3),
    }

    # ── 7. Build JSON payload ────────────────────────────────────────────────
    centerline = [
        {
            "frame_index": int(points_raw[i]["frame_index"]),
            "timestamp_sec": round(float(points_raw[i]["timestamp_sec"]), 6),
            "x": round(float(xs[i]), 3),
            "y": round(float(ys[i]), 3),
        }
        for i in range(n)
    ]
    left_edge = [{"x": round(float(lx[i]), 3), "y": round(float(ly[i]), 3)} for i in range(n)]
    right_edge = [{"x": round(float(rx[i]), 3), "y": round(float(ry[i]), 3)} for i in range(n)]
    corridor_polygon = [{"x": round(float(poly_x[i]), 3), "y": round(float(poly_y[i]), 3)} for i in range(len(poly_x))]

    corridor_payload: dict[str, Any] = {
        "expert_code": expert_code,
        "source_model": payload.get("source_model", "yolo_sam2"),
        "source_trajectory": smoothed_path.name,
        "corridor_type": "normal_offset_polyline",
        "margin_px": margin_px,
        "centerline_point_count": n,
        "centerline": centerline,
        "left_edge": left_edge,
        "right_edge": right_edge,
        "corridor_polygon": corridor_polygon,
        "bbox": bbox,
        "quality": {
            "all_points_finite": all_finite,
            "polygon_closed": polygon_closed,
            "corridor_area_px2": round(corridor_area, 2),
            "mean_width_px": margin_px * 2,
            "warnings": warnings,
        },
        "usable_for_learner_comparison": all_finite and polygon_closed and n >= 20,
    }

    corridor_path = out_dir / CORRIDOR_FILENAME
    _write_json(corridor_path, corridor_payload)

    # ── 8. Preview PNG ───────────────────────────────────────────────────────
    canvas_w, canvas_h = _canvas_size_from_summary(out_dir)
    preview_path = out_dir / CORRIDOR_PREVIEW_FILENAME
    _render_corridor_preview(
        canvas_w=canvas_w,
        canvas_h=canvas_h,
        centerline=centerline,
        left_edge=left_edge,
        right_edge=right_edge,
        corridor_polygon=corridor_polygon,
        output_path=preview_path,
    )

    # ── 9. Corridor overlay video ────────────────────────────────────────────
    overlay_path = out_dir / CORRIDOR_OVERLAY_FILENAME
    _render_corridor_overlay_video(
        out_dir=out_dir,
        centerline=centerline,
        left_edge=left_edge,
        right_edge=right_edge,
        corridor_polygon=corridor_polygon,
        output_path=overlay_path,
    )

    # ── 10. Stamp summary.json / metadata.json ───────────────────────────────
    for stub_name in ("summary.json", "metadata.json"):
        stub_path = out_dir / stub_name
        if stub_path.is_file():
            stub = json.loads(stub_path.read_text(encoding="utf-8"))
            stub["expert_corridor_json_path"] = str(corridor_path)
            stub["expert_corridor_preview_path"] = str(preview_path)
            stub["expert_corridor_overlay_path"] = str(overlay_path)
            stub["expert_corridor_margin_px"] = margin_px
            _write_json(stub_path, stub)

    # ── 11. Print summary ────────────────────────────────────────────────────
    print(f"[corridor] expert_code        : {expert_code}")
    print(f"[corridor] centerline points  : {n}")
    print(f"[corridor] margin_px          : {margin_px}")
    print(f"[corridor] corridor_area_px2  : {corridor_area:.1f}")
    print(f"[corridor] bbox               : {bbox}")
    print(f"[corridor] warnings           : {warnings or 'none'}")
    print(f"[corridor] output             : {corridor_path}")
    print(f"[corridor] preview            : {preview_path}")
    print(f"[corridor] overlay_video      : {overlay_path}")
    return corridor_payload


# ──────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ──────────────────────────────────────────────────────────────────────────────

def _polygon_area(pts: np.ndarray) -> float:
    """Shoelace formula for polygon area."""
    x = pts[:, 0]
    y = pts[:, 1]
    return 0.5 * float(np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1))))


def _canvas_size_from_summary(out_dir: Path) -> tuple[int, int]:
    """Try to derive (width, height) from summary.json or raw.json metadata."""
    summary_path = out_dir / "summary.json"
    if summary_path.is_file():
        try:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            video_path = summary.get("video_path")
            if video_path:
                cap = cv2.VideoCapture(video_path)
                if cap.isOpened():
                    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    cap.release()
                    if w > 0 and h > 0:
                        return w, h
        except Exception:  # noqa: BLE001
            pass
    return 1920, 1080


def _pts_to_int32(edges: list[dict[str, Any]]) -> np.ndarray:
    return np.array([[int(round(p["x"])), int(round(p["y"]))] for p in edges], dtype=np.int32)


def _render_corridor_preview(
    *,
    canvas_w: int,
    canvas_h: int,
    centerline: list[dict],
    left_edge: list[dict],
    right_edge: list[dict],
    corridor_polygon: list[dict],
    output_path: Path,
) -> None:
    canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
    _draw_corridor(canvas, corridor_polygon, left_edge, right_edge, centerline)
    cv2.imwrite(str(output_path), canvas)


def _draw_corridor(
    frame: np.ndarray,
    corridor_polygon: list[dict],
    left_edge: list[dict],
    right_edge: list[dict],
    centerline: list[dict],
) -> None:
    """Draw corridor onto an existing BGR frame in-place."""
    poly_pts = _pts_to_int32(corridor_polygon)
    overlay = frame.copy()
    cv2.fillPoly(overlay, [poly_pts], _COLOR_CORRIDOR)
    cv2.addWeighted(overlay, _POLY_ALPHA, frame, 1 - _POLY_ALPHA, 0, dst=frame)

    left_pts = _pts_to_int32(left_edge)
    right_pts = _pts_to_int32(right_edge)
    center_pts = np.array(
        [[int(round(p["x"])), int(round(p["y"]))] for p in centerline], dtype=np.int32
    )

    cv2.polylines(frame, [left_pts], isClosed=False, color=_COLOR_LEFT, thickness=2)
    cv2.polylines(frame, [right_pts], isClosed=False, color=_COLOR_RIGHT, thickness=2)
    cv2.polylines(frame, [center_pts], isClosed=False, color=_COLOR_CENTERLINE, thickness=2)

    if len(center_pts) > 0:
        cv2.circle(frame, tuple(center_pts[0]), 10, _COLOR_START, -1)
        cv2.circle(frame, tuple(center_pts[-1]), 10, _COLOR_END, -1)
        cv2.putText(frame, "START", (center_pts[0][0] + 12, center_pts[0][1]),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, _COLOR_START, 2)
        cv2.putText(frame, "END", (center_pts[-1][0] + 12, center_pts[-1][1]),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, _COLOR_END, 2)


def _render_corridor_overlay_video(
    *,
    out_dir: Path,
    centerline: list[dict],
    left_edge: list[dict],
    right_edge: list[dict],
    corridor_polygon: list[dict],
    output_path: Path,
) -> None:
    """Write corridor_overlay.mp4 using the original expert video as background."""
    summary_path = out_dir / "summary.json"
    source_video: str | None = None
    fps: float = 30.0
    if summary_path.is_file():
        try:
            s = json.loads(summary_path.read_text(encoding="utf-8"))
            source_video = s.get("video_path")
        except Exception:  # noqa: BLE001
            pass

    cap: cv2.VideoCapture | None = None
    if source_video:
        cap = cv2.VideoCapture(source_video)
        if not cap.isOpened():
            cap = None

    if cap is not None:
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    else:
        w, h = 1920, 1080

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (w, h))

    if cap is not None:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            _draw_corridor(frame, corridor_polygon, left_edge, right_edge, centerline)
            writer.write(frame)
        cap.release()
    else:
        # No source video: blank black frames, one per centerline point
        blank = np.zeros((h, w, 3), dtype=np.uint8)
        for _ in range(max(len(centerline), 30)):
            frame = blank.copy()
            _draw_corridor(frame, corridor_polygon, left_edge, right_edge, centerline)
            writer.write(frame)

    writer.release()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")
