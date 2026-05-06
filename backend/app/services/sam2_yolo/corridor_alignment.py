"""Corridor alignment for learner YOLO+SAM2 trajectory.

Post-processing step after trajectory_smoothed.json is created.
Default alignment uses frame-0 SAM2 mask bbox anchors from raw.json.

Called automatically by runner.py after a successful learner run when
expert_code is provided.
"""

from __future__ import annotations

import json
import logging
import math
from pathlib import Path
from typing import Any
from urllib.parse import quote

import cv2
import numpy as np

logger = logging.getLogger(__name__)

BACKEND_ROOT = Path(__file__).resolve().parents[3]

ALIGNED_CORRIDOR_FILENAME = "aligned_corridor.json"
ALIGNED_CORRIDOR_FRAME0_PREVIEW_FILENAME = "aligned_corridor_preview.png"
ALIGNED_CORRIDOR_PROGRESS_FILENAME = "aligned_corridor_progress.json"
ALIGNED_CORRIDOR_PREVIEW_FILENAME = "aligned_corridor_progress_preview.png"
ALIGNED_CORRIDOR_OVERLAY_FILENAME = "sam2_yolo_aligned_corridor_overlay.mp4"

# BGR drawing colours
_COLOR_CORRIDOR_FILL = (0, 180, 0)      # green fill
_COLOR_CENTERLINE = (255, 255, 255)     # white
_COLOR_EDGE = (0, 255, 255)             # yellow (BGR)
_COLOR_LEARNER = (255, 255, 0)          # cyan (BGR)
_COLOR_OUTSIDE_PT = (0, 0, 255)         # red (BGR)
_COLOR_INSIDE_PT = (0, 255, 0)          # green (BGR)
_COLOR_EXPERT_START = (0, 255, 0)       # green
_COLOR_LEARNER_START = (255, 0, 0)      # blue (BGR)
_CORRIDOR_ALPHA = 0.3


# ──────────────────────────────────────────────────────────────────────────────
# Public API — default: blade-tip translation + robust extension
# ──────────────────────────────────────────────────────────────────────────────

# Segment filtering thresholds
_MIN_SEGMENT_LENGTH_PX = 80.0
_MIN_NET_DISPLACEMENT_PX = 60.0
_HEADING_THRESHOLD_DEG = 45.0
_SUSTAINED_WINDOW_SEGS = 5
_TANGENT_WINDOW_SEGS = 8
_STRAIGHT_LINE_CODES: set[str] = {"straight_line_v1"}


def align_corridor_blade_tip_with_extension(
    *,
    corridor_path: str,
    expert_raw_path: str,       # not used; kept for runner compatibility
    learner_raw_path: str,      # not used; kept for runner compatibility
    learner_smoothed_path: str,
    learner_run_id: str,
    output_dir: str,
    expert_code: str,
    learner_video_path: str | None = None,
) -> dict[str, Any]:
    """Align pre-built expert corridor via blade-tip translation + safe axis extension.

    Step 1  Pure translation: expert_anchor = corridor centerline[0],
            learner_anchor = smoothed[0].  Entire corridor geometry translated
            as a rigid body — no normals rebuilt, no shape changed.

    Step 2  Robust segmentation of the translated centerline.
            Segments shorter than _MIN_SEGMENT_LENGTH_PX or with net
            displacement < _MIN_NET_DISPLACEMENT_PX are REJECTED and never
            extended (prevents 36-px noise tail → 489-px horizontal corridor bug).
            For straight_line_v1 a single global-axis segment is forced.

    Step 3  For each ACCEPTED segment, project learner points onto the expert
            segment axis (forward + lateral-proximity filtered).  If the learner
            advanced further than the segment length, extend ALONG THE EXPERT
            AXIS ONLY.  Following segments are shifted by the extension vector
            to stay continuous.

    Step 4  Deviation check: each learner point vs. nearest adapted centerline
            point → perpendicular signed deviation.
    """
    out_dir = Path(output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    corridor = json.loads(Path(corridor_path).read_text(encoding="utf-8"))
    smoothed = json.loads(Path(learner_smoothed_path).read_text(encoding="utf-8"))

    # ── Step 1: anchors & translation ─────────────────────────────────────
    expert_anchor = (
        float(corridor["centerline"][0]["x"]),
        float(corridor["centerline"][0]["y"]),
    )
    smoothed_pts = smoothed.get("points", [])
    if not smoothed_pts:
        raise ValueError("trajectory_smoothed.json has no points")
    first_pt = smoothed_pts[0]
    learner_anchor = (float(first_pt["smoothed_x"]), float(first_pt["smoothed_y"]))
    t_dx = learner_anchor[0] - expert_anchor[0]
    t_dy = learner_anchor[1] - expert_anchor[1]

    print(f"[corridor_alignment] Expert anchor: ({expert_anchor[0]:.2f}, {expert_anchor[1]:.2f})", flush=True)
    print(f"[corridor_alignment] Learner anchor: ({learner_anchor[0]:.2f}, {learner_anchor[1]:.2f})", flush=True)
    print(f"[corridor_alignment] Translation: dx={t_dx:.2f}, dy={t_dy:.2f}", flush=True)

    margin_px = float(corridor["margin_px"])

    def _translate(pts_list: list[dict[str, Any]]) -> list[tuple[float, float]]:
        return [(float(p["x"]) + t_dx, float(p["y"]) + t_dy) for p in pts_list]

    aligned_cl = _translate(corridor["centerline"])
    aligned_left = _translate(corridor["left_edge"])
    aligned_right = _translate(corridor["right_edge"])
    original_n = len(aligned_cl)

    dist0 = math.sqrt(
        (aligned_cl[0][0] - learner_anchor[0]) ** 2
        + (aligned_cl[0][1] - learner_anchor[1]) ** 2
    )
    print(f"[corridor_alignment] Distance aligned_start → learner_anchor: {dist0:.2f} px", flush=True)
    print(f"[corridor_alignment] Original centerline points: {original_n}", flush=True)

    # ── Step 2: robust segmentation ───────────────────────────────────────
    learner_points: list[tuple[float, float]] = [
        (float(p["smoothed_x"]), float(p["smoothed_y"]))
        for p in smoothed_pts
        if p.get("smoothed_x") is not None and p.get("smoothed_y") is not None
    ]
    print(f"[corridor_alignment] Learner points: {len(learner_points)}", flush=True)

    accepted_segs, rejected_segs = _robust_segment_centerline(aligned_cl, expert_code)

    print(f"[corridor_alignment] Accepted segments: {len(accepted_segs)}", flush=True)
    print(f"[corridor_alignment] Rejected segments: {len(rejected_segs)}", flush=True)
    if expert_code in _STRAIGHT_LINE_CODES and len(accepted_segs) > 1:
        print(
            f"[corridor_alignment] WARNING: {expert_code} produced {len(accepted_segs)} "
            "accepted segments — expected 1.",
            flush=True,
        )
    for rs in rejected_segs:
        print(
            f"[corridor_alignment]   REJECTED seg{rs['segment_index']}: "
            f"length={rs['original_length_px']:.1f}px  "
            f"net_disp={rs['net_displacement_px']:.1f}px  "
            f"reason={rs['rejection_reason']}",
            flush=True,
        )

    # ── Step 3: extend accepted segments along expert axes ─────────────────
    adapted_cl, adapted_left, adapted_right, adapted_seg_info, ext_debug = (
        _apply_robust_extension(
            aligned_cl, aligned_left, aligned_right,
            accepted_segs, learner_points, margin_px,
        )
    )

    # Sanity check: no rejected segment was accidentally extended
    for rs in rejected_segs:
        if rs.get("extension_px", 0.0) > 0:
            raise RuntimeError(
                f"BUG: rejected segment {rs['segment_index']} was extended. "
                "This should never happen."
            )

    adapted_polygon = adapted_left + list(reversed(adapted_right)) + [adapted_left[0]]
    adapted_n = len(adapted_cl)
    print(f"[corridor_alignment] Adapted centerline points: {adapted_n}", flush=True)

    # ── Width validation ───────────────────────────────────────────────────
    width_debug = _compute_width_debug(adapted_left, adapted_right, margin_px)
    print(
        f"[corridor_alignment] Width validation: "
        f"expected={width_debug['expected_width_px']:.1f}px  "
        f"min={width_debug['min_width_px']:.1f}px  "
        f"max={width_debug['max_width_px']:.1f}px  "
        f"mean={width_debug['mean_width_px']:.1f}px",
        flush=True,
    )
    _tol = 0.15 * width_debug["expected_width_px"]
    if (
        width_debug["min_width_px"] < width_debug["expected_width_px"] - _tol
        or width_debug["max_width_px"] > width_debug["expected_width_px"] + _tol
    ):
        print(
            "[corridor_alignment] WARNING: corridor width distortion detected. "
            f"min={width_debug['min_width_px']:.1f}px "
            f"max={width_debug['max_width_px']:.1f}px "
            f"expected={width_debug['expected_width_px']:.1f}px",
            flush=True,
        )

    # ── Step 4: deviation check ────────────────────────────────────────────
    _, _, normals, _ = _build_corridor_edges(adapted_cl, margin_px)
    frame_checks = _compute_nearest_checks(
        learner_points, smoothed_pts, adapted_cl, normals, margin_px
    )

    outside_count = sum(1 for c in frame_checks if c["outside"])
    total = len(frame_checks)
    devs = [c["deviation_px"] for c in frame_checks]
    max_dev = max(devs) if devs else 0.0
    mean_dev = sum(devs) / max(len(devs), 1)
    outside_pct = 100.0 * outside_count / max(total, 1)

    print(f"[corridor_alignment] Outside frames: {outside_count} / {total}", flush=True)
    print(f"[corridor_alignment] Max deviation: {max_dev:.2f} px", flush=True)
    print(f"[corridor_alignment] Mean deviation: {mean_dev:.2f} px", flush=True)
    print(f"[corridor_alignment] Outside percentage: {outside_pct:.1f}%", flush=True)
    if outside_pct > 80.0:
        print(
            "[corridor_alignment] WARNING: learner path mostly outside corridor. "
            "Check anchor, margin, or trajectory quality.",
            flush=True,
        )

    # ── Save JSON ──────────────────────────────────────────────────────────
    aligned_json_path = out_dir / ALIGNED_CORRIDOR_FILENAME
    payload: dict[str, Any] = {
        "expert_id": expert_code,
        "learner_run_id": learner_run_id,
        "alignment_mode": "blade_tip_translation_with_robust_extension",
        "anchor_mode": "expert_corridor_start_to_learner_smoothed_blade_tip_start",
        "margin_px": margin_px,
        "expert_anchor": {"x": round(expert_anchor[0], 2), "y": round(expert_anchor[1], 2)},
        "learner_anchor": {"x": round(learner_anchor[0], 2), "y": round(learner_anchor[1], 2)},
        "translation": {"dx": round(t_dx, 2), "dy": round(t_dy, 2)},
        "original_centerline_points": original_n,
        "adapted_centerline_point_count": adapted_n,
        "learner_points_count": len(learner_points),
        "expert_segment_count": len(accepted_segs),
        "expert_segments": adapted_seg_info,
        "rejected_segments": rejected_segs,
        "extension_debug": ext_debug,
        "width_debug": width_debug,
        "aligned_centerline": [
            {"x": round(x, 2), "y": round(y, 2)} for x, y in adapted_cl
        ],
        "left_edge": [{"x": round(x, 2), "y": round(y, 2)} for x, y in adapted_left],
        "right_edge": [{"x": round(x, 2), "y": round(y, 2)} for x, y in adapted_right],
        "corridor_polygon": [
            {"x": round(x, 2), "y": round(y, 2)} for x, y in adapted_polygon
        ],
        "frame_checks": frame_checks,
        "summary": {
            "total_frames_checked": total,
            "outside_frames": outside_count,
            "inside_frames": total - outside_count,
            "max_deviation_px": round(max_dev, 2),
            "mean_deviation_px": round(mean_dev, 2),
            "outside_percentage": round(outside_pct, 1),
        },
    }
    _write_json(aligned_json_path, payload)

    # ── Preview PNG ────────────────────────────────────────────────────────
    preview_path = out_dir / ALIGNED_CORRIDOR_FRAME0_PREVIEW_FILENAME
    canvas_w, canvas_h = _video_dimensions(learner_video_path)
    _render_extended_preview(
        canvas_w=canvas_w,
        canvas_h=canvas_h,
        adapted_cl=adapted_cl,
        adapted_left=adapted_left,
        adapted_right=adapted_right,
        adapted_polygon=adapted_polygon,
        learner_points=learner_points,
        frame_checks=frame_checks,
        learner_anchor=learner_anchor,
        t_dx=t_dx,
        t_dy=t_dy,
        margin_px=margin_px,
        outside_pct=outside_pct,
        adapted_seg_info=adapted_seg_info,
        rejected_segs=rejected_segs,
        output_path=preview_path,
        width_debug=width_debug,
    )

    # ── Overlay video ──────────────────────────────────────────────────────
    overlay_path = out_dir / ALIGNED_CORRIDOR_OVERLAY_FILENAME
    _render_blade_tip_overlay(
        learner_video_path=learner_video_path,
        aligned_centerline=adapted_cl,
        aligned_polygon=adapted_polygon,
        aligned_left_edge=adapted_left,
        aligned_right_edge=adapted_right,
        learner_points=learner_points,
        frame_checks=frame_checks,
        learner_anchor=learner_anchor,
        output_path=overlay_path,
    )

    print(f"[corridor_alignment] Aligned corridor JSON: {aligned_json_path}", flush=True)
    print(f"[corridor_alignment] Preview image: {preview_path}", flush=True)
    print(f"[corridor_alignment] Overlay video: {overlay_path}", flush=True)

    return {
        "aligned_corridor_json_path": str(aligned_json_path),
        "aligned_corridor_preview_path": str(preview_path),
        "aligned_corridor_preview_url": _storage_url_for(preview_path),
        "aligned_corridor_overlay_video_path": str(overlay_path),
        "aligned_corridor_overlay_video_url": _storage_url_for(overlay_path),
    }


# ── Robust segmentation helpers ────────────────────────────────────────────────

def _compute_width_debug(
    left_edge: list[tuple[float, float]],
    right_edge: list[tuple[float, float]],
    margin_px: float,
) -> dict[str, float]:
    """Compute per-sample corridor width and return summary statistics.

    Expected width = 2 * margin_px.  Width is measured as the Euclidean
    distance between corresponding left_edge[i] and right_edge[i] points.
    The original translated edges preserve the expert width exactly; only
    extension points may differ slightly if the axis-derived normal differs
    from the original corridor normal near the tail.
    """
    expected = 2.0 * margin_px
    n = min(len(left_edge), len(right_edge))
    if n == 0:
        return {
            "expected_width_px": round(expected, 2),
            "min_width_px": 0.0,
            "max_width_px": 0.0,
            "mean_width_px": 0.0,
        }
    widths = [
        math.sqrt(
            (left_edge[i][0] - right_edge[i][0]) ** 2
            + (left_edge[i][1] - right_edge[i][1]) ** 2
        )
        for i in range(n)
    ]
    return {
        "expected_width_px": round(expected, 2),
        "min_width_px": round(min(widths), 2),
        "max_width_px": round(max(widths), 2),
        "mean_width_px": round(sum(widths) / n, 2),
    }


def _robust_segment_centerline(
    centerline: list[tuple[float, float]],
    expert_code: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Return (accepted_segments, rejected_segments) with filtering.

    Segments shorter than _MIN_SEGMENT_LENGTH_PX or with net displacement <
    _MIN_NET_DISPLACEMENT_PX are rejected and flagged.  For straight-line
    expert codes, a single global-axis segment is forced unconditionally.
    """
    n = len(centerline)
    if n < 2:
        seg = _make_expert_segment(0, n - 1, centerline, force_straight=True)
        return [_seg_to_accepted(seg, 0)], []

    if expert_code in _STRAIGHT_LINE_CODES:
        seg = _make_expert_segment(0, n - 1, centerline, force_straight=True)
        info = _seg_to_accepted(seg, 0)
        print(
            f"[corridor_alignment] {expert_code}: forced 1 global-axis segment "
            f"(length={info['original_length_px']:.1f}px  "
            f"dir=({info['pca_direction']['x']:.3f},{info['pca_direction']['y']:.3f}))",
            flush=True,
        )
        return [info], []

    raw_segs = _detect_expert_segments(
        centerline,
        heading_threshold_deg=_HEADING_THRESHOLD_DEG,
        sustained_window=_SUSTAINED_WINDOW_SEGS,
        tangent_window=_TANGENT_WINDOW_SEGS,
    )

    accepted: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    for idx, seg in enumerate(raw_segs):
        pts = seg["pts"]
        length = seg["length_px"]
        net_disp = (
            math.sqrt((pts[-1][0] - pts[0][0]) ** 2 + (pts[-1][1] - pts[0][1]) ** 2)
            if len(pts) >= 2
            else 0.0
        )
        reasons: list[str] = []
        if length < _MIN_SEGMENT_LENGTH_PX:
            reasons.append(f"length {length:.1f}px < {_MIN_SEGMENT_LENGTH_PX}px")
        if net_disp < _MIN_NET_DISPLACEMENT_PX:
            reasons.append(f"net_disp {net_disp:.1f}px < {_MIN_NET_DISPLACEMENT_PX}px")

        if reasons:
            rejected.append({
                "segment_index": idx,
                "start_index": seg["start_index"],
                "end_index": seg["end_index"],
                "original_length_px": round(length, 2),
                "net_displacement_px": round(net_disp, 2),
                "rejection_reason": "; ".join(reasons),
                "extension_px": 0.0,
            })
        else:
            accepted.append(_seg_to_accepted(seg, idx))

    return accepted, rejected


def _seg_to_accepted(seg: dict[str, Any], idx: int) -> dict[str, Any]:
    sx, sy = seg["pca_direction"]
    return {
        "segment_index": idx,
        "start_index": seg["start_index"],
        "end_index": seg["end_index"],
        "original_length_px": seg["length_px"],
        "pca_direction": {"x": round(sx, 4), "y": round(sy, 4)},
        "is_straight": seg["is_straight"],
        "is_curved": seg["is_curved"],
    }


def _apply_robust_extension(
    aligned_cl: list[tuple[float, float]],
    aligned_left: list[tuple[float, float]],
    aligned_right: list[tuple[float, float]],
    accepted_segs: list[dict[str, Any]],
    learner_points: list[tuple[float, float]],
    margin_px: float,
) -> tuple[
    list[tuple[float, float]],
    list[tuple[float, float]],
    list[tuple[float, float]],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    """Extend accepted segments along expert axes; shift following segments to stay continuous.

    Original corridor geometry is kept intact for each segment (just shifted
    by cumulative prior extensions).  Only extension points are newly computed.

    Returns (adapted_cl, adapted_left, adapted_right, adapted_seg_info, ext_debug).
    """
    if not accepted_segs:
        return aligned_cl[:], aligned_left[:], aligned_right[:], [], []

    adapted_cl: list[tuple[float, float]] = []
    adapted_left: list[tuple[float, float]] = []
    adapted_right: list[tuple[float, float]] = []
    adapted_seg_info: list[dict[str, Any]] = []
    ext_debug: list[dict[str, Any]] = []
    cumulative_shift = (0.0, 0.0)

    for seg in accepted_segs:
        s = seg["start_index"]
        e = seg["end_index"]
        sx = seg["pca_direction"]["x"]
        sy = seg["pca_direction"]["y"]
        cs_x, cs_y = cumulative_shift

        # Original geometry for this segment, shifted by prior extensions
        seg_cl = [(aligned_cl[i][0] + cs_x, aligned_cl[i][1] + cs_y) for i in range(s, e + 1)]
        seg_left = [(aligned_left[i][0] + cs_x, aligned_left[i][1] + cs_y) for i in range(s, e + 1)]
        seg_right = [(aligned_right[i][0] + cs_x, aligned_right[i][1] + cs_y) for i in range(s, e + 1)]

        # Deduplicate junction point with previous segment
        if adapted_cl:
            seg_cl = seg_cl[1:]
            seg_left = seg_left[1:]
            seg_right = seg_right[1:]

        adapted_cl.extend(seg_cl)
        adapted_left.extend(seg_left)
        adapted_right.extend(seg_right)

        # Forward projection of learner points onto this segment's expert axis
        seg_start = (aligned_cl[s][0] + cs_x, aligned_cl[s][1] + cs_y)
        perp_limit = 3.0 * margin_px
        max_proj = 0.0
        for lx, ly in learner_points:
            fwd = (lx - seg_start[0]) * sx + (ly - seg_start[1]) * sy
            if fwd <= 0:
                continue
            perp = abs((lx - seg_start[0]) * (-sy) + (ly - seg_start[1]) * sx)
            if perp > perp_limit:
                continue
            if fwd > max_proj:
                max_proj = fwd

        original_length = seg["original_length_px"]
        extension_px = max(0.0, max_proj - original_length)
        ext_vec = (0.0, 0.0)

        if extension_px > 1.0:
            end_cl_pt = adapted_cl[-1]
            end_left_pt = adapted_left[-1]

            # Normal direction: perpendicular to PCA axis, consistent with existing edges
            nx, ny = -sy, sx
            chk_dx = end_left_pt[0] - end_cl_pt[0]
            chk_dy = end_left_pt[1] - end_cl_pt[1]
            if chk_dx * nx + chk_dy * ny < 0:
                nx, ny = -nx, -ny

            n_extra = max(1, int(round(extension_px / 5)))
            step_size = extension_px / n_extra
            for k in range(1, n_extra + 1):
                ext_cl = (
                    end_cl_pt[0] + sx * step_size * k,
                    end_cl_pt[1] + sy * step_size * k,
                )
                adapted_cl.append(ext_cl)
                adapted_left.append((ext_cl[0] + margin_px * nx, ext_cl[1] + margin_px * ny))
                adapted_right.append((ext_cl[0] - margin_px * nx, ext_cl[1] - margin_px * ny))

            ext_vec = (sx * extension_px, sy * extension_px)
            print(
                f"[corridor_alignment]   seg{seg['segment_index']}: "
                f"extended {extension_px:.1f}px along ({sx:.3f},{sy:.3f})",
                flush=True,
            )

        cumulative_shift = (cumulative_shift[0] + ext_vec[0], cumulative_shift[1] + ext_vec[1])

        adapted_seg_info.append({
            "segment_index": seg["segment_index"],
            "start_index": s,
            "end_index": e,
            "is_straight": seg["is_straight"],
            "is_curved": seg["is_curved"],
            "original_length_px": round(original_length, 2),
            "adapted_length_px": round(original_length + extension_px, 2),
            "direction_vector": {"x": round(sx, 4), "y": round(sy, 4)},
            "extension_mode": "global_expert_axis_extension" if extension_px > 1.0 else "none",
            "extension_px": round(extension_px, 2),
            "extension_vector": {"x": round(ext_vec[0], 2), "y": round(ext_vec[1], 2)},
        })
        ext_debug.append({
            "segment_index": seg["segment_index"],
            "max_learner_projection_px": round(max_proj, 2),
            "original_length_px": round(original_length, 2),
            "extension_px": round(extension_px, 2),
            "perp_limit_px": round(perp_limit, 2),
        })

    return adapted_cl, adapted_left, adapted_right, adapted_seg_info, ext_debug


def _render_extended_preview(
    *,
    canvas_w: int,
    canvas_h: int,
    adapted_cl: list[tuple[float, float]],
    adapted_left: list[tuple[float, float]],
    adapted_right: list[tuple[float, float]],
    adapted_polygon: list[tuple[float, float]],
    learner_points: list[tuple[float, float]],
    frame_checks: list[dict[str, Any]],
    learner_anchor: tuple[float, float],
    t_dx: float,
    t_dy: float,
    margin_px: float,
    outside_pct: float,
    adapted_seg_info: list[dict[str, Any]],
    rejected_segs: list[dict[str, Any]],
    output_path: Path,
    width_debug: dict[str, float] | None = None,
    bbox_center_debug: tuple[float, float] | None = None,
) -> None:
    """Render the aligned corridor preview image.

    Visual layers (bottom → top):
      1. Transparent green filled corridor polygon — expert reference
      2. Yellow thin edges (left/right)
      3. Cyan/blue expert centerline
      4. White learner smoothed blade-tip trajectory (separate from corridor)
      5. Green dots — learner inside points
         Red dots   — learner outside points
      6. Blue filled circle — learner blade-tip anchor
         Gray circle (optional) — raw bbox_center debug marker
      7. Info text block (top-left)
    """
    canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)

    # 1. Corridor polygon — transparent green fill
    if len(adapted_polygon) >= 3:
        ov = canvas.copy()
        cv2.fillPoly(ov, [_pts_i32(adapted_polygon)], _COLOR_CORRIDOR_FILL)
        cv2.addWeighted(ov, _CORRIDOR_ALPHA, canvas, 1 - _CORRIDOR_ALPHA, 0, dst=canvas)

    # 2. Yellow corridor edges (thin)
    if len(adapted_left) >= 2:
        cv2.polylines(canvas, [_pts_i32(adapted_left)], False, _COLOR_EDGE, 1)
    if len(adapted_right) >= 2:
        cv2.polylines(canvas, [_pts_i32(adapted_right)], False, _COLOR_EDGE, 1)

    # 3. Expert centerline — cyan
    if len(adapted_cl) >= 2:
        cv2.polylines(canvas, [_pts_i32(adapted_cl)], False, _COLOR_ADAPTED_CL, 2)

    # 4. Learner smoothed blade-tip trajectory — white (separate from corridor)
    if len(learner_points) >= 2:
        cv2.polylines(canvas, [_pts_i32(learner_points)], False, _COLOR_CENTERLINE, 2)

    # 5. Per-frame deviation dots
    for chk in frame_checks:
        pt = (int(round(chk["learner_x"])), int(round(chk["learner_y"])))
        if chk["outside"]:
            cv2.circle(canvas, pt, 6, _COLOR_OUTSIDE_PT, -1)   # red — outside
        else:
            cv2.circle(canvas, pt, 3, _COLOR_INSIDE_PT, -1)    # green — inside

    # 6a. Optional gray bbox_center debug marker
    if bbox_center_debug is not None:
        bc_pt = (int(round(bbox_center_debug[0])), int(round(bbox_center_debug[1])))
        cv2.circle(canvas, bc_pt, 8, (120, 120, 120), 2)
        cv2.putText(
            canvas, "bbox_center",
            (bc_pt[0] + 10, bc_pt[1] + 5),
            cv2.FONT_HERSHEY_SIMPLEX, 0.42, (120, 120, 120), 1, cv2.LINE_AA,
        )

    # 6b. Learner blade-tip anchor — solid blue circle
    la_pt = (int(round(learner_anchor[0])), int(round(learner_anchor[1])))
    cv2.circle(canvas, la_pt, 10, _COLOR_LEARNER_START, -1)     # filled blue
    cv2.circle(canvas, la_pt, 10, (255, 255, 255), 1)            # thin white outline
    cv2.putText(
        canvas, "LEARNER BLADE TIP START",
        (la_pt[0] + 14, la_pt[1] + 5),
        cv2.FONT_HERSHEY_SIMPLEX, 0.52, _COLOR_LEARNER_START, 2, cv2.LINE_AA,
    )

    # 7. Info text block
    wd = width_debug or {}
    width_ok = True
    if wd:
        _tol = 0.15 * wd.get("expected_width_px", 1)
        width_ok = (
            wd.get("min_width_px", 0) >= wd.get("expected_width_px", 0) - _tol
            and wd.get("max_width_px", 0) <= wd.get("expected_width_px", 0) + _tol
        )

    info: list[str] = [
        f"mode: blade_tip+robust_extension  margin={int(margin_px)}px  "
        f"expected_width={int(2*margin_px)}px",
        f"dx={t_dx:.1f}  dy={t_dy:.1f}  outside={outside_pct:.1f}%",
    ]
    if wd:
        w_flag = "" if width_ok else "  ⚠ WIDTH DISTORTION"
        info.append(
            f"width: min={wd.get('min_width_px', 0):.1f}  "
            f"max={wd.get('max_width_px', 0):.1f}  "
            f"mean={wd.get('mean_width_px', 0):.1f}px{w_flag}"
        )
    info.append(f"accepted_segs={len(adapted_seg_info)}  rejected={len(rejected_segs)}")
    for seg in adapted_seg_info:
        info.append(
            f"  seg{seg['segment_index']}: "
            f"{seg['original_length_px']:.0f}→{seg['adapted_length_px']:.0f}px "
            f"dir=({seg['direction_vector']['x']:.3f},{seg['direction_vector']['y']:.3f}) "
            f"[{seg['extension_mode']}]"
        )
    for rs in rejected_segs:
        info.append(
            f"  REJECTED seg{rs['segment_index']}: {rs['original_length_px']:.0f}px "
            f"({rs['rejection_reason']})"
        )
    for li, txt in enumerate(info):
        cv2.putText(
            canvas, txt,
            (18, 30 + li * 24),
            cv2.FONT_HERSHEY_SIMPLEX, 0.50, _COLOR_CENTERLINE, 1, cv2.LINE_AA,
        )

    cv2.imwrite(str(output_path), canvas)


# ── Archive: blade-tip pure translation, no extension (kept for reference) ────

def align_corridor_blade_tip_translation(
    *,
    corridor_path: str,
    expert_raw_path: str,       # not used; kept for runner compatibility
    learner_raw_path: str,      # not used; kept for runner compatibility
    learner_smoothed_path: str,
    learner_run_id: str,
    output_dir: str,
    expert_code: str,
    learner_video_path: str | None = None,
) -> dict[str, Any]:
    """Align the pre-built expert corridor to the learner video using pure translation.

    Anchors
    -------
    expert_anchor  = corridor["centerline"][0]              (blade-tip first point)
    learner_anchor = trajectory_smoothed["points"][0]       (blade-tip first point)

    The entire corridor geometry (centerline, left_edge, right_edge,
    corridor_polygon) is translated by (dx, dy).  Normals are NOT rebuilt from
    scratch; the translated polygon preserves the original corridor shape exactly.

    No segmentation.  No dynamic extension.  No normal recomputation.
    """
    out_dir = Path(output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    corridor = json.loads(Path(corridor_path).read_text(encoding="utf-8"))
    smoothed = json.loads(Path(learner_smoothed_path).read_text(encoding="utf-8"))

    # ── Anchors ──────────────────────────────────────────────────────────
    expert_anchor = (
        float(corridor["centerline"][0]["x"]),
        float(corridor["centerline"][0]["y"]),
    )
    smoothed_pts = smoothed.get("points", [])
    if not smoothed_pts:
        raise ValueError("trajectory_smoothed.json has no points")
    first_pt = smoothed_pts[0]
    learner_anchor = (float(first_pt["smoothed_x"]), float(first_pt["smoothed_y"]))
    dx = learner_anchor[0] - expert_anchor[0]
    dy = learner_anchor[1] - expert_anchor[1]

    print(f"[corridor_alignment] Expert anchor: ({expert_anchor[0]:.2f}, {expert_anchor[1]:.2f})", flush=True)
    print(f"[corridor_alignment] Learner anchor: ({learner_anchor[0]:.2f}, {learner_anchor[1]:.2f})", flush=True)
    print(f"[corridor_alignment] Translation: dx={dx:.2f}, dy={dy:.2f}", flush=True)

    margin_px = float(corridor["margin_px"])

    # ── Translate corridor geometry as a rigid body ────────────────────
    def _translate(pts_list: list[dict[str, Any]]) -> list[tuple[float, float]]:
        return [(float(p["x"]) + dx, float(p["y"]) + dy) for p in pts_list]

    aligned_centerline = _translate(corridor["centerline"])
    aligned_left_edge = _translate(corridor["left_edge"])
    aligned_right_edge = _translate(corridor["right_edge"])
    aligned_polygon = _translate(corridor["corridor_polygon"])

    dist0 = math.sqrt(
        (aligned_centerline[0][0] - learner_anchor[0]) ** 2
        + (aligned_centerline[0][1] - learner_anchor[1]) ** 2
    )
    original_n = len(aligned_centerline)
    print(f"[corridor_alignment] Distance aligned_start → learner_anchor: {dist0:.2f} px", flush=True)
    print(f"[corridor_alignment] Original centerline points: {original_n}", flush=True)

    # ── Learner smoothed points ────────────────────────────────────────
    learner_points: list[tuple[float, float]] = [
        (float(p["smoothed_x"]), float(p["smoothed_y"]))
        for p in smoothed_pts
        if p.get("smoothed_x") is not None and p.get("smoothed_y") is not None
    ]
    print(f"[corridor_alignment] Learner points: {len(learner_points)}", flush=True)

    # Compute normals from translated centerline (same directions as original;
    # translation preserves all angles). Used only for signed-deviation math.
    _, _, normals, _ = _build_corridor_edges(aligned_centerline, margin_px)

    # ── Per-frame deviation check ──────────────────────────────────────
    frame_checks = _compute_nearest_checks(
        learner_points, smoothed_pts, aligned_centerline, normals, margin_px
    )

    outside_count = sum(1 for c in frame_checks if c["outside"])
    total = len(frame_checks)
    devs = [c["deviation_px"] for c in frame_checks]
    max_dev = max(devs) if devs else 0.0
    mean_dev = sum(devs) / max(len(devs), 1)
    outside_pct = 100.0 * outside_count / max(total, 1)

    print(f"[corridor_alignment] Outside frames: {outside_count} / {total}", flush=True)
    print(f"[corridor_alignment] Max deviation: {max_dev:.2f} px", flush=True)
    print(f"[corridor_alignment] Mean deviation: {mean_dev:.2f} px", flush=True)
    print(f"[corridor_alignment] Outside percentage: {outside_pct:.1f}%", flush=True)
    if outside_pct > 80.0:
        print(
            "[corridor_alignment] WARNING: learner path mostly outside corridor. "
            "Check anchor, margin, or trajectory quality.",
            flush=True,
        )

    # ── Save aligned_corridor.json ────────────────────────────────────
    aligned_json_path = out_dir / ALIGNED_CORRIDOR_FILENAME
    payload: dict[str, Any] = {
        "expert_id": expert_code,
        "learner_run_id": learner_run_id,
        "alignment_mode": "blade_tip_corridor_translation_only",
        "anchor_mode": "expert_corridor_start_to_learner_smoothed_blade_tip_start",
        "margin_px": margin_px,
        "expert_anchor": {"x": round(expert_anchor[0], 2), "y": round(expert_anchor[1], 2)},
        "learner_anchor": {"x": round(learner_anchor[0], 2), "y": round(learner_anchor[1], 2)},
        "translation": {"dx": round(dx, 2), "dy": round(dy, 2)},
        "original_centerline_points": original_n,
        "adapted_centerline_point_count": original_n,
        "learner_points_count": len(learner_points),
        "aligned_centerline": [
            {"x": round(x, 2), "y": round(y, 2)} for x, y in aligned_centerline
        ],
        "left_edge": [{"x": round(x, 2), "y": round(y, 2)} for x, y in aligned_left_edge],
        "right_edge": [{"x": round(x, 2), "y": round(y, 2)} for x, y in aligned_right_edge],
        "corridor_polygon": [
            {"x": round(x, 2), "y": round(y, 2)} for x, y in aligned_polygon
        ],
        "frame_checks": frame_checks,
        "summary": {
            "total_frames_checked": total,
            "outside_frames": outside_count,
            "inside_frames": total - outside_count,
            "max_deviation_px": round(max_dev, 2),
            "mean_deviation_px": round(mean_dev, 2),
            "outside_percentage": round(outside_pct, 1),
        },
    }
    _write_json(aligned_json_path, payload)

    # ── Preview PNG ───────────────────────────────────────────────────
    preview_path = out_dir / ALIGNED_CORRIDOR_FRAME0_PREVIEW_FILENAME
    canvas_w, canvas_h = _video_dimensions(learner_video_path)
    _render_blade_tip_preview(
        canvas_w=canvas_w,
        canvas_h=canvas_h,
        aligned_centerline=aligned_centerline,
        aligned_left_edge=aligned_left_edge,
        aligned_right_edge=aligned_right_edge,
        aligned_polygon=aligned_polygon,
        learner_points=learner_points,
        frame_checks=frame_checks,
        learner_anchor=learner_anchor,
        dx=dx,
        dy=dy,
        margin_px=margin_px,
        outside_pct=outside_pct,
        output_path=preview_path,
    )

    # ── Overlay video ─────────────────────────────────────────────────
    overlay_path = out_dir / ALIGNED_CORRIDOR_OVERLAY_FILENAME
    _render_blade_tip_overlay(
        learner_video_path=learner_video_path,
        aligned_centerline=aligned_centerline,
        aligned_polygon=aligned_polygon,
        aligned_left_edge=aligned_left_edge,
        aligned_right_edge=aligned_right_edge,
        learner_points=learner_points,
        frame_checks=frame_checks,
        learner_anchor=learner_anchor,
        output_path=overlay_path,
    )

    print(f"[corridor_alignment] Aligned corridor JSON: {aligned_json_path}", flush=True)
    print(f"[corridor_alignment] Preview image: {preview_path}", flush=True)
    print(f"[corridor_alignment] Overlay video: {overlay_path}", flush=True)

    return {
        "aligned_corridor_json_path": str(aligned_json_path),
        "aligned_corridor_preview_path": str(preview_path),
        "aligned_corridor_preview_url": _storage_url_for(preview_path),
        "aligned_corridor_overlay_video_path": str(overlay_path),
        "aligned_corridor_overlay_video_url": _storage_url_for(overlay_path),
    }


# ── Blade-tip translation rendering ───────────────────────────────────────────

def _render_blade_tip_preview(
    *,
    canvas_w: int,
    canvas_h: int,
    aligned_centerline: list[tuple[float, float]],
    aligned_left_edge: list[tuple[float, float]],
    aligned_right_edge: list[tuple[float, float]],
    aligned_polygon: list[tuple[float, float]],
    learner_points: list[tuple[float, float]],
    frame_checks: list[dict[str, Any]],
    learner_anchor: tuple[float, float],
    dx: float,
    dy: float,
    margin_px: float,
    outside_pct: float,
    output_path: Path,
) -> None:
    canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)

    ov = canvas.copy()
    cv2.fillPoly(ov, [_pts_i32(aligned_polygon)], _COLOR_CORRIDOR_FILL)
    cv2.addWeighted(ov, _CORRIDOR_ALPHA, canvas, 1 - _CORRIDOR_ALPHA, 0, dst=canvas)

    if len(aligned_left_edge) >= 2:
        cv2.polylines(canvas, [_pts_i32(aligned_left_edge)], False, _COLOR_EDGE, 1)
    if len(aligned_right_edge) >= 2:
        cv2.polylines(canvas, [_pts_i32(aligned_right_edge)], False, _COLOR_EDGE, 1)
    if len(aligned_centerline) >= 2:
        cv2.polylines(canvas, [_pts_i32(aligned_centerline)], False, _COLOR_ADAPTED_CL, 2)
    if len(learner_points) >= 2:
        cv2.polylines(canvas, [_pts_i32(learner_points)], False, _COLOR_CENTERLINE, 2)

    for chk in frame_checks:
        pt = (int(round(chk["learner_x"])), int(round(chk["learner_y"])))
        if chk["outside"]:
            cv2.circle(canvas, pt, 6, _COLOR_OUTSIDE_PT, -1)
        else:
            cv2.circle(canvas, pt, 3, _COLOR_INSIDE_PT, -1)

    la_pt = (int(round(learner_anchor[0])), int(round(learner_anchor[1])))
    cv2.circle(canvas, la_pt, 14, (255, 255, 255), 2)
    cv2.circle(canvas, la_pt, 7, _COLOR_LEARNER_START, -1)
    cv2.putText(
        canvas, "ALIGNED EXPERT START",
        (la_pt[0] + 17, la_pt[1] - 8),
        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA,
    )
    cv2.putText(
        canvas, "LEARNER BLADE TIP START",
        (la_pt[0] + 17, la_pt[1] + 18),
        cv2.FONT_HERSHEY_SIMPLEX, 0.55, _COLOR_LEARNER_START, 2, cv2.LINE_AA,
    )

    info = [
        f"mode: blade_tip_corridor_translation_only  margin={int(margin_px)}px",
        f"dx={dx:.1f}  dy={dy:.1f}  outside={outside_pct:.1f}%",
        "corridor: expert geometry translated — no extension, no normal rebuild",
    ]
    for li, txt in enumerate(info):
        cv2.putText(
            canvas, txt,
            (18, 30 + li * 26),
            cv2.FONT_HERSHEY_SIMPLEX, 0.58, _COLOR_CENTERLINE, 2, cv2.LINE_AA,
        )

    cv2.imwrite(str(output_path), canvas)


def _render_blade_tip_overlay(
    *,
    learner_video_path: str | None,
    aligned_centerline: list[tuple[float, float]],
    aligned_polygon: list[tuple[float, float]],
    aligned_left_edge: list[tuple[float, float]],
    aligned_right_edge: list[tuple[float, float]],
    learner_points: list[tuple[float, float]],
    frame_checks: list[dict[str, Any]],
    learner_anchor: tuple[float, float],
    output_path: Path,
) -> None:
    cap: cv2.VideoCapture | None = None
    fps, w, h = 30.0, 1920, 1080

    if learner_video_path:
        cap = cv2.VideoCapture(str(learner_video_path))
        if not cap.isOpened():
            cap = None
    if cap is not None:
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)

    writer = cv2.VideoWriter(
        str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h)
    )

    def _draw(frame: np.ndarray) -> None:
        ov = frame.copy()
        cv2.fillPoly(ov, [_pts_i32(aligned_polygon)], _COLOR_CORRIDOR_FILL)
        cv2.addWeighted(ov, _CORRIDOR_ALPHA, frame, 1 - _CORRIDOR_ALPHA, 0, dst=frame)
        if len(aligned_left_edge) >= 2:
            cv2.polylines(frame, [_pts_i32(aligned_left_edge)], False, _COLOR_EDGE, 1)
        if len(aligned_right_edge) >= 2:
            cv2.polylines(frame, [_pts_i32(aligned_right_edge)], False, _COLOR_EDGE, 1)
        if len(aligned_centerline) >= 2:
            cv2.polylines(frame, [_pts_i32(aligned_centerline)], False, _COLOR_ADAPTED_CL, 2)
        if len(learner_points) >= 2:
            cv2.polylines(frame, [_pts_i32(learner_points)], False, _COLOR_CENTERLINE, 2)
        for chk in frame_checks:
            pt = (int(round(chk["learner_x"])), int(round(chk["learner_y"])))
            cv2.circle(
                frame, pt,
                6 if chk["outside"] else 3,
                _COLOR_OUTSIDE_PT if chk["outside"] else _COLOR_INSIDE_PT,
                -1,
            )
        la_pt = (int(round(learner_anchor[0])), int(round(learner_anchor[1])))
        cv2.circle(frame, la_pt, 10, _COLOR_EXPERT_ANCHOR, 2)
        cv2.circle(frame, la_pt, 5, _COLOR_LEARNER_ANCHOR_PT, -1)

    try:
        if cap is not None:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                _draw(frame)
                writer.write(frame)
            cap.release()
        else:
            blank = np.zeros((h, w, 3), dtype=np.uint8)
            for _ in range(max(len(aligned_centerline), 30)):
                frame = blank.copy()
                _draw(frame)
                writer.write(frame)
    finally:
        writer.release()


# ── Archive: expert-axis-preserving alignment (kept for reference) ────────────

def align_corridor_expert_axis(
    *,
    corridor_path: str,
    expert_raw_path: str,  # not used for anchor; kept for runner compatibility
    learner_raw_path: str,
    learner_smoothed_path: str,
    learner_run_id: str,
    output_dir: str,
    expert_code: str,
    learner_video_path: str | None = None,
) -> dict[str, Any]:
    """Align expert corridor to the learner video while preserving expert geometry.

    Anchor placement
    ----------------
    Expert anchor  = corridor["centerline"][0]  (first expert corridor point)
    Learner anchor = learner_raw frames[0]["bbox_center"]
    A pure translation moves the expert corridor start to the learner anchor.

    Shape preservation
    ------------------
    The expert corridor shape is NEVER modified by learner movement.
    Each expert geometric segment may be EXTENDED along its own expert axis if
    the learner progressed further along that direction — but the direction
    itself stays fixed.  Subsequent segments are shifted to keep continuity.
    """
    out_dir = Path(output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    corridor = json.loads(Path(corridor_path).read_text(encoding="utf-8"))
    learner_raw = json.loads(Path(learner_raw_path).read_text(encoding="utf-8"))
    smoothed = json.loads(Path(learner_smoothed_path).read_text(encoding="utf-8"))

    # ── Step 2: anchor & translation ─────────────────────────────────────
    learner_frame0 = _frame0_payload(learner_raw, "learner")
    learner_anchor = _xy_tuple(learner_frame0.get("bbox_center"), "learner frame 0 bbox_center")

    expert_cl_raw = corridor["centerline"]
    expert_cl_start = (float(expert_cl_raw[0]["x"]), float(expert_cl_raw[0]["y"]))

    translation = (
        learner_anchor[0] - expert_cl_start[0],
        learner_anchor[1] - expert_cl_start[1],
    )
    margin_px = float(corridor["margin_px"])

    # Translate the whole expert centerline
    expert_cl: list[tuple[float, float]] = [
        (float(p["x"]) + translation[0], float(p["y"]) + translation[1])
        for p in expert_cl_raw
    ]

    dist0 = math.sqrt(
        (expert_cl[0][0] - learner_anchor[0]) ** 2
        + (expert_cl[0][1] - learner_anchor[1]) ** 2
    )
    print(f"[corridor_alignment] Expert corridor start: ({expert_cl_start[0]:.2f}, {expert_cl_start[1]:.2f})", flush=True)
    print(f"[corridor_alignment] Learner bbox center anchor: ({learner_anchor[0]:.2f}, {learner_anchor[1]:.2f})", flush=True)
    print(f"[corridor_alignment] Translation: dx={translation[0]:.2f}, dy={translation[1]:.2f}", flush=True)
    print(f"[corridor_alignment] Distance aligned_start → learner_anchor: {dist0:.2f} px", flush=True)

    # ── Step 4: segment the translated expert centerline ─────────────────
    # Experts known to be a single straight cut bypass heading-change segmentation
    # so that noise in the last few centerline points cannot create spurious short
    # segments that then get extended in the wrong (horizontal/diagonal) direction.
    _STRAIGHT_LINE_CODES = {"straight_line_v1"}
    if expert_code in _STRAIGHT_LINE_CODES:
        segments = [_make_expert_segment(0, len(expert_cl) - 1, expert_cl, force_straight=True)]
        print(f"[corridor_alignment] Straight-line expert '{expert_code}': forcing single global-axis segment", flush=True)
    else:
        segments = _detect_expert_segments(expert_cl)
    print(f"[corridor_alignment] Expert segment count: {len(segments)}", flush=True)

    # ── Load learner smoothed points ─────────────────────────────────────
    smoothed_pts_raw = smoothed.get("points", [])
    learner_points: list[tuple[float, float]] = [
        (float(p["smoothed_x"]), float(p["smoothed_y"]))
        for p in smoothed_pts_raw
        if p.get("smoothed_x") is not None and p.get("smoothed_y") is not None
    ]
    if not learner_points:
        raise ValueError("trajectory_smoothed.json has no valid smoothed points")

    # ── Steps 5-6: extend segments along expert directions only ──────────
    adapted_centerline, adapted_seg_info = _adapt_expert_segments(
        expert_cl, segments, learner_points, margin_px
    )

    original_n = len(expert_cl)
    adapted_n = len(adapted_centerline)
    print(f"[corridor_alignment] Original centerline points: {original_n}", flush=True)
    print(f"[corridor_alignment] Adapted centerline points: {adapted_n}", flush=True)
    print(f"[corridor_alignment] Learner points: {len(learner_points)}", flush=True)
    for seg in adapted_seg_info:
        print(
            f"[corridor_alignment]   Segment {seg['segment_index']}: "
            f"orig={seg['original_length_px']:.1f}px  "
            f"adapted={seg['adapted_length_px']:.1f}px  "
            f"dir=({seg['direction_vector']['x']:.3f},{seg['direction_vector']['y']:.3f})  "
            f"mode={seg['extension_mode']}",
            flush=True,
        )

    # ── Step 8: rebuild corridor polygon with fixed margin_px ────────────
    left_edge, right_edge, normals, corridor_polygon = _build_corridor_edges(
        adapted_centerline, margin_px
    )

    # ── Step 9: per-frame learner deviation check ─────────────────────────
    frame_checks = _compute_nearest_checks(
        learner_points, smoothed_pts_raw, adapted_centerline, normals, margin_px
    )

    outside_count = sum(1 for c in frame_checks if c["outside"])
    total = len(frame_checks)
    devs = [c["deviation_px"] for c in frame_checks]
    max_dev = max(devs) if devs else 0.0
    mean_dev = sum(devs) / max(len(devs), 1)
    outside_pct = 100.0 * outside_count / max(total, 1)

    print(f"[corridor_alignment] Outside frames: {outside_count} / {total}", flush=True)
    print(f"[corridor_alignment] Max deviation: {max_dev:.2f} px", flush=True)
    print(f"[corridor_alignment] Mean deviation: {mean_dev:.2f} px", flush=True)
    print(f"[corridor_alignment] Outside percentage: {outside_pct:.1f}%", flush=True)
    if outside_pct > 80.0:
        print(
            "[corridor_alignment] WARNING: learner path mostly outside corridor. "
            "Check anchor, margin, or trajectory quality.",
            flush=True,
        )

    # ── Step 10: save aligned_corridor.json ──────────────────────────────
    aligned_json_path = out_dir / ALIGNED_CORRIDOR_FILENAME
    payload: dict[str, Any] = {
        "expert_id": expert_code,
        "learner_run_id": learner_run_id,
        "alignment_mode": "bbox_center_to_expert_centerline_start_translation",
        "margin_px": margin_px,
        "learner_anchor_bbox_center": {
            "x": round(learner_anchor[0], 2),
            "y": round(learner_anchor[1], 2),
        },
        "expert_centerline_start": {
            "x": round(expert_cl_start[0], 2),
            "y": round(expert_cl_start[1], 2),
        },
        "translation": {"dx": round(translation[0], 2), "dy": round(translation[1], 2)},
        "expert_segment_count": len(adapted_seg_info),
        "expert_segments": adapted_seg_info,
        "original_centerline_points": original_n,
        "adapted_centerline_point_count": adapted_n,
        "learner_points_count": len(learner_points),
        "aligned_centerline": [
            {"x": round(x, 2), "y": round(y, 2)} for x, y in adapted_centerline
        ],
        "left_edge": [{"x": round(x, 2), "y": round(y, 2)} for x, y in left_edge],
        "right_edge": [{"x": round(x, 2), "y": round(y, 2)} for x, y in right_edge],
        "corridor_polygon": [
            {"x": round(x, 2), "y": round(y, 2)} for x, y in corridor_polygon
        ],
        "frame_checks": frame_checks,
        "summary": {
            "total_frames_checked": total,
            "outside_frames": outside_count,
            "inside_frames": total - outside_count,
            "max_deviation_px": round(max_dev, 2),
            "mean_deviation_px": round(mean_dev, 2),
            "outside_percentage": round(outside_pct, 1),
        },
    }
    _write_json(aligned_json_path, payload)

    # ── Step 11: preview PNG ─────────────────────────────────────────────
    preview_path = out_dir / ALIGNED_CORRIDOR_FRAME0_PREVIEW_FILENAME
    canvas_w, canvas_h = _video_dimensions(learner_video_path)
    _render_expert_axis_preview(
        canvas_w=canvas_w,
        canvas_h=canvas_h,
        adapted_centerline=adapted_centerline,
        expert_cl=expert_cl,
        left_edge=left_edge,
        right_edge=right_edge,
        corridor_polygon=corridor_polygon,
        learner_points=learner_points,
        frame_checks=frame_checks,
        learner_anchor=learner_anchor,
        margin_px=margin_px,
        outside_pct=outside_pct,
        adapted_seg_info=adapted_seg_info,
        output_path=preview_path,
    )

    # Overlay video
    overlay_path = out_dir / ALIGNED_CORRIDOR_OVERLAY_FILENAME
    _render_expert_axis_overlay(
        learner_video_path=learner_video_path,
        adapted_centerline=adapted_centerline,
        corridor_polygon=corridor_polygon,
        left_edge=left_edge,
        right_edge=right_edge,
        learner_points=learner_points,
        frame_checks=frame_checks,
        learner_anchor=learner_anchor,
        output_path=overlay_path,
    )

    print(f"[corridor_alignment] Aligned corridor JSON: {aligned_json_path}", flush=True)
    print(f"[corridor_alignment] Preview image: {preview_path}", flush=True)
    print(f"[corridor_alignment] Overlay video: {overlay_path}", flush=True)

    return {
        "aligned_corridor_json_path": str(aligned_json_path),
        "aligned_corridor_preview_path": str(preview_path),
        "aligned_corridor_preview_url": _storage_url_for(preview_path),
        "aligned_corridor_overlay_video_path": str(overlay_path),
        "aligned_corridor_overlay_video_url": _storage_url_for(overlay_path),
    }


# ── Expert-axis helpers ────────────────────────────────────────────────────────

def _pca_direction(points: list[tuple[float, float]]) -> tuple[float, float]:
    """PCA main axis direction, oriented from first toward last point."""
    if len(points) < 2:
        return (0.0, -1.0)
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    cx = sum(xs) / len(xs)
    cy = sum(ys) / len(ys)
    sxx = sum((x - cx) ** 2 for x in xs)
    syy = sum((y - cy) ** 2 for y in ys)
    sxy = sum((x - cx) * (y - cy) for x, y in zip(xs, ys))
    tr = sxx + syy
    det = sxx * syy - sxy * sxy
    disc = math.sqrt(max(0.0, (tr / 2) ** 2 - det))
    lam1 = tr / 2 + disc
    vx, vy = lam1 - syy, sxy
    mag = math.sqrt(vx * vx + vy * vy)
    if mag < 1e-10:
        vx, vy = sxy, lam1 - sxx
        mag = math.sqrt(vx * vx + vy * vy)
    if mag < 1e-10:
        dx = points[-1][0] - points[0][0]
        dy = points[-1][1] - points[0][1]
        m = math.sqrt(dx * dx + dy * dy)
        return (dx / m, dy / m) if m > 0 else (0.0, -1.0)
    vx, vy = vx / mag, vy / mag
    end_dx = points[-1][0] - points[0][0]
    end_dy = points[-1][1] - points[0][1]
    if vx * end_dx + vy * end_dy < 0:
        vx, vy = -vx, -vy
    return (vx, vy)


def _heading_angle_diff(a: float, b: float) -> float:
    """Signed angle difference in [-pi, pi]."""
    d = a - b
    while d > math.pi:
        d -= 2 * math.pi
    while d < -math.pi:
        d += 2 * math.pi
    return d


def _detect_expert_segments(
    centerline: list[tuple[float, float]],
    heading_threshold_deg: float = 40.0,
    sustained_window: int = 4,
    tangent_window: int = 5,
) -> list[dict[str, Any]]:
    """Split translated expert centerline into geometric segments.

    Uses local tangent angles and detects sustained heading changes.
    For paths with no turns (e.g. straight_line_v1) this returns a single
    segment whose direction is the global PCA axis.
    """
    n = len(centerline)
    if n < max(2 * tangent_window + 2, 10):
        return [_make_expert_segment(0, n - 1, centerline)]

    angles: list[float] = []
    for i in range(n):
        i0 = max(0, i - tangent_window)
        i1 = min(n - 1, i + tangent_window)
        dx = centerline[i1][0] - centerline[i0][0]
        dy = centerline[i1][1] - centerline[i0][1]
        angles.append(math.atan2(dy, dx))

    threshold_rad = math.radians(heading_threshold_deg)
    split_indices: list[int] = []
    i = sustained_window
    while i < n - sustained_window:
        diff = abs(_heading_angle_diff(angles[i], angles[i - sustained_window]))
        if diff > threshold_rad:
            # Walk forward to find peak of the heading change
            peak = i
            for j in range(i + 1, min(i + sustained_window * 2, n - sustained_window)):
                if abs(_heading_angle_diff(angles[j], angles[j - sustained_window])) > abs(
                    _heading_angle_diff(angles[peak], angles[peak - sustained_window])
                ):
                    peak = j
            split_indices.append(peak)
            i = peak + sustained_window
        else:
            i += 1

    boundaries_s = [0] + split_indices
    boundaries_e = split_indices + [n - 1]
    return [
        _make_expert_segment(s, e, centerline)
        for s, e in zip(boundaries_s, boundaries_e)
    ]


def _make_expert_segment(
    start: int,
    end: int,
    centerline: list[tuple[float, float]],
    force_straight: bool = False,
) -> dict[str, Any]:
    """Compute properties for one geometric segment of the expert centerline.

    force_straight=True is used when the caller has already decided this is a
    straight segment (e.g. straight_line_v1 single-segment bypass).  It
    guarantees is_straight=True so PCA direction is always used for extension.
    """
    pts = centerline[start: end + 1]
    length = sum(
        math.sqrt(
            (pts[j + 1][0] - pts[j][0]) ** 2 + (pts[j + 1][1] - pts[j][1]) ** 2
        )
        for j in range(len(pts) - 1)
    ) if len(pts) > 1 else 0.0
    if len(pts) >= 2:
        dx = pts[-1][0] - pts[0][0]
        dy = pts[-1][1] - pts[0][1]
        m = math.sqrt(dx * dx + dy * dy)
        raw_dir: tuple[float, float] = (dx / m, dy / m) if m > 0 else (0.0, -1.0)
    else:
        raw_dir = (0.0, -1.0)
    pca_dir = _pca_direction(pts)
    # Determine straight vs curved via angular spread unless caller forces straight
    if force_straight:
        is_straight = True
    else:
        win = 5
        seg_angles = []
        for i in range(len(pts)):
            i0, i1 = max(0, i - win), min(len(pts) - 1, i + win)
            ddx = pts[i1][0] - pts[i0][0]
            ddy = pts[i1][1] - pts[i0][1]
            seg_angles.append(math.atan2(ddy, ddx))
        spread = (
            max(abs(_heading_angle_diff(a, seg_angles[0])) for a in seg_angles)
            if len(seg_angles) > 1
            else 0.0
        )
        is_straight = spread < math.radians(15)
    return {
        "start_index": start,
        "end_index": end,
        "pts": pts,
        "length_px": length,
        "pca_direction": pca_dir,
        "raw_direction": raw_dir,
        "is_straight": is_straight,
        "is_curved": not is_straight,
    }


def _adapt_expert_segments(
    expert_cl: list[tuple[float, float]],
    segments: list[dict[str, Any]],
    learner_points: list[tuple[float, float]],
    margin_px: float,
) -> tuple[list[tuple[float, float]], list[dict[str, Any]]]:
    """Extend expert segments along expert axes; shift following segments to stay continuous.

    For each segment:
    - Use PCA direction (straight) or raw endpoint direction (curved) as the axis.
    - Project learner points onto the axis (forward projections only, filtered by
      lateral proximity to avoid cross-segment contamination).
    - If a learner point projects further than the segment length, extend along
      the EXPERT axis direction by the surplus.
    - Accumulate extension vectors so all subsequent segments stay connected.
    """
    cumulative_shift = (0.0, 0.0)
    adapted_parts: list[list[tuple[float, float]]] = []
    adapted_seg_info: list[dict[str, Any]] = []

    for seg_idx, seg in enumerate(segments):
        # Apply cumulative extension from previous segments
        pts = [
            (x + cumulative_shift[0], y + cumulative_shift[1])
            for x, y in seg["pts"]
        ]

        # Use PCA for straight (immune to endpoint noise), raw for curved
        sx, sy = seg["pca_direction"] if seg["is_straight"] else seg["raw_direction"]
        seg_start = pts[0]
        perp_limit = 3.0 * margin_px  # prevents cross-segment contamination

        max_proj = 0.0
        for lx, ly in learner_points:
            fwd = (lx - seg_start[0]) * sx + (ly - seg_start[1]) * sy
            if fwd <= 0:
                continue
            perp = abs((lx - seg_start[0]) * (-sy) + (ly - seg_start[1]) * sx)
            if perp > perp_limit:
                continue
            if fwd > max_proj:
                max_proj = fwd

        original_length = seg["length_px"]
        extension_px = max(0.0, max_proj - original_length)

        if extension_px > 1.0:
            end_pt = pts[-1]
            n_extra = max(1, int(round(extension_px / 5)))
            step = extension_px / n_extra
            ext_pts = [
                (end_pt[0] + sx * step * k, end_pt[1] + sy * step * k)
                for k in range(1, n_extra + 1)
            ]
            adapted_pts = pts + ext_pts
            ext_vec = (sx * extension_px, sy * extension_px)
            mode = "global_expert_axis_extension" if seg["is_straight"] else "arc_length_preserve_curve"
        else:
            adapted_pts = pts
            ext_vec = (0.0, 0.0)
            extension_px = 0.0
            mode = "none"

        adapted_parts.append(adapted_pts)
        cumulative_shift = (
            cumulative_shift[0] + ext_vec[0],
            cumulative_shift[1] + ext_vec[1],
        )
        adapted_seg_info.append({
            "segment_index": seg_idx,
            "start_index": seg["start_index"],
            "end_index": seg["end_index"],
            "is_straight": seg["is_straight"],
            "is_curved": seg["is_curved"],
            "original_length_px": round(original_length, 2),
            "adapted_length_px": round(original_length + extension_px, 2),
            "direction_vector": {"x": round(sx, 4), "y": round(sy, 4)},
            "extension_mode": mode,
            "extension_px": round(extension_px, 2),
            "extension_vector": {"x": round(ext_vec[0], 2), "y": round(ext_vec[1], 2)},
        })

    # Concatenate parts, deduplicating junction points
    adapted_cl: list[tuple[float, float]] = []
    for part in adapted_parts:
        if adapted_cl:
            adapted_cl.extend(part[1:])
        else:
            adapted_cl.extend(part)
    return adapted_cl, adapted_seg_info


# Expert-axis rendering colours
_COLOR_ADAPTED_CL = (0, 200, 255)         # cyan (BGR) for adapted expert centerline
_COLOR_ORIG_EXPERT_CL = (60, 60, 180)     # dim blue for pre-extension expert line


def _draw_expert_axis_corridor(
    frame: np.ndarray,
    adapted_centerline: list[tuple[float, float]],
    corridor_polygon: list[tuple[float, float]],
    left_edge: list[tuple[float, float]],
    right_edge: list[tuple[float, float]],
    learner_points: list[tuple[float, float]],
    frame_checks: list[dict[str, Any]],
    learner_anchor: tuple[float, float],
) -> None:
    ov = frame.copy()
    cv2.fillPoly(ov, [_pts_i32(corridor_polygon)], _COLOR_CORRIDOR_FILL)
    cv2.addWeighted(ov, _CORRIDOR_ALPHA, frame, 1 - _CORRIDOR_ALPHA, 0, dst=frame)
    if len(left_edge) >= 2:
        cv2.polylines(frame, [_pts_i32(left_edge)], False, _COLOR_EDGE, 1)
    if len(right_edge) >= 2:
        cv2.polylines(frame, [_pts_i32(right_edge)], False, _COLOR_EDGE, 1)
    if len(adapted_centerline) >= 2:
        cv2.polylines(frame, [_pts_i32(adapted_centerline)], False, _COLOR_ADAPTED_CL, 2)
    if len(learner_points) >= 2:
        cv2.polylines(frame, [_pts_i32(learner_points)], False, _COLOR_CENTERLINE, 2)
    for chk in frame_checks:
        pt = (int(round(chk["learner_x"])), int(round(chk["learner_y"])))
        cv2.circle(
            frame, pt,
            6 if chk["outside"] else 3,
            _COLOR_OUTSIDE_PT if chk["outside"] else _COLOR_INSIDE_PT,
            -1,
        )
    la_pt = (int(round(learner_anchor[0])), int(round(learner_anchor[1])))
    cv2.circle(frame, la_pt, 10, _COLOR_EXPERT_ANCHOR, 2)
    cv2.circle(frame, la_pt, 5, _COLOR_LEARNER_ANCHOR_PT, -1)


def _render_expert_axis_preview(
    *,
    canvas_w: int,
    canvas_h: int,
    adapted_centerline: list[tuple[float, float]],
    expert_cl: list[tuple[float, float]],
    left_edge: list[tuple[float, float]],
    right_edge: list[tuple[float, float]],
    corridor_polygon: list[tuple[float, float]],
    learner_points: list[tuple[float, float]],
    frame_checks: list[dict[str, Any]],
    learner_anchor: tuple[float, float],
    margin_px: float,
    outside_pct: float,
    adapted_seg_info: list[dict[str, Any]],
    output_path: Path,
) -> None:
    canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)

    ov = canvas.copy()
    cv2.fillPoly(ov, [_pts_i32(corridor_polygon)], _COLOR_CORRIDOR_FILL)
    cv2.addWeighted(ov, _CORRIDOR_ALPHA, canvas, 1 - _CORRIDOR_ALPHA, 0, dst=canvas)

    if len(left_edge) >= 2:
        cv2.polylines(canvas, [_pts_i32(left_edge)], False, _COLOR_EDGE, 1)
    if len(right_edge) >= 2:
        cv2.polylines(canvas, [_pts_i32(right_edge)], False, _COLOR_EDGE, 1)
    # Original translated expert centerline (dim blue) — before segment extension
    if len(expert_cl) >= 2:
        cv2.polylines(canvas, [_pts_i32(expert_cl)], False, _COLOR_ORIG_EXPERT_CL, 1)
    # Adapted (potentially extended) centerline in cyan
    if len(adapted_centerline) >= 2:
        cv2.polylines(canvas, [_pts_i32(adapted_centerline)], False, _COLOR_ADAPTED_CL, 2)
    # Learner trajectory in white (distinct from corridor)
    if len(learner_points) >= 2:
        cv2.polylines(canvas, [_pts_i32(learner_points)], False, _COLOR_CENTERLINE, 2)

    for chk in frame_checks:
        pt = (int(round(chk["learner_x"])), int(round(chk["learner_y"])))
        if chk["outside"]:
            cv2.circle(canvas, pt, 6, _COLOR_OUTSIDE_PT, -1)
        else:
            cv2.circle(canvas, pt, 3, _COLOR_INSIDE_PT, -1)

    # Anchors coincide by construction — draw as concentric rings
    la_pt = (int(round(learner_anchor[0])), int(round(learner_anchor[1])))
    cv2.circle(canvas, la_pt, 14, (255, 255, 255), 2)
    cv2.circle(canvas, la_pt, 7, _COLOR_LEARNER_START, -1)
    cv2.putText(
        canvas, "ALIGNED EXPERT START",
        (la_pt[0] + 17, la_pt[1] - 8),
        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA,
    )
    cv2.putText(
        canvas, "LEARNER BBOX CENTER",
        (la_pt[0] + 17, la_pt[1] + 18),
        cv2.FONT_HERSHEY_SIMPLEX, 0.55, _COLOR_LEARNER_START, 2, cv2.LINE_AA,
    )

    info: list[str] = [
        f"mode: expert_axis_translation  margin={int(margin_px)}px",
        f"segments: {len(adapted_seg_info)}  outside: {outside_pct:.1f}%",
    ]
    for seg in adapted_seg_info:
        info.append(
            f"  seg{seg['segment_index']}: "
            f"{seg['original_length_px']:.0f}→{seg['adapted_length_px']:.0f}px "
            f"dir=({seg['direction_vector']['x']:.3f},{seg['direction_vector']['y']:.3f}) "
            f"[{seg['extension_mode']}]"
        )
    for li, txt in enumerate(info):
        cv2.putText(
            canvas, txt,
            (18, 30 + li * 26),
            cv2.FONT_HERSHEY_SIMPLEX, 0.58, _COLOR_CENTERLINE, 2, cv2.LINE_AA,
        )

    cv2.imwrite(str(output_path), canvas)


def _render_expert_axis_overlay(
    *,
    learner_video_path: str | None,
    adapted_centerline: list[tuple[float, float]],
    corridor_polygon: list[tuple[float, float]],
    left_edge: list[tuple[float, float]],
    right_edge: list[tuple[float, float]],
    learner_points: list[tuple[float, float]],
    frame_checks: list[dict[str, Any]],
    learner_anchor: tuple[float, float],
    output_path: Path,
) -> None:
    cap: cv2.VideoCapture | None = None
    fps, w, h = 30.0, 1920, 1080

    if learner_video_path:
        cap = cv2.VideoCapture(str(learner_video_path))
        if not cap.isOpened():
            cap = None
    if cap is not None:
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)

    writer = cv2.VideoWriter(
        str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h)
    )

    def _draw(frame: np.ndarray) -> None:
        _draw_expert_axis_corridor(
            frame, adapted_centerline, corridor_polygon,
            left_edge, right_edge, learner_points, frame_checks, learner_anchor,
        )

    try:
        if cap is not None:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                _draw(frame)
                writer.write(frame)
            cap.release()
        else:
            blank = np.zeros((h, w, 3), dtype=np.uint8)
            for _ in range(max(len(adapted_centerline), 30)):
                frame = blank.copy()
                _draw(frame)
                writer.write(frame)
    finally:
        writer.release()


# ── Archive: translation-only (kept for reference, not the default) ──────────

def align_corridor_translation_only(
    *,
    corridor_path: str,
    expert_raw_path: str,
    learner_raw_path: str,
    learner_smoothed_path: str,
    learner_run_id: str,
    output_dir: str,
    expert_code: str,
    learner_video_path: str | None = None,
) -> dict[str, Any]:
    """Align expert corridor to learner using frame-0 bbox_center translation only.

    No scaling, no rotation.  The corridor width stays fixed from the expert.
    The centerline is extrapolated at both ends to cover the full learner Y
    range.  Per-smoothed-point deviation checks use nearest-centerline lookup.

    Returns dict with paths and public storage URLs.
    """
    out_dir = Path(output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    corridor = json.loads(Path(corridor_path).read_text(encoding="utf-8"))
    expert_raw = json.loads(Path(expert_raw_path).read_text(encoding="utf-8"))
    learner_raw = json.loads(Path(learner_raw_path).read_text(encoding="utf-8"))
    smoothed = json.loads(Path(learner_smoothed_path).read_text(encoding="utf-8"))

    # ── Step 1: anchors & translation ─────────────────────────────────────
    expert_frame0 = _frame0_payload(expert_raw, "expert")
    learner_frame0 = _frame0_payload(learner_raw, "learner")
    expert_anchor = _xy_tuple(expert_frame0.get("bbox_center"), "expert frame 0 bbox_center")
    learner_anchor = _xy_tuple(learner_frame0.get("bbox_center"), "learner frame 0 bbox_center")

    translation = (
        learner_anchor[0] - expert_anchor[0],
        learner_anchor[1] - expert_anchor[1],
    )
    print(f"[corridor_alignment] Expert anchor: ({expert_anchor[0]:.2f}, {expert_anchor[1]:.2f})", flush=True)
    print(f"[corridor_alignment] Learner anchor: ({learner_anchor[0]:.2f}, {learner_anchor[1]:.2f})", flush=True)
    print(f"[corridor_alignment] Translation: dx={translation[0]:.2f}, dy={translation[1]:.2f}", flush=True)

    # ── Step 2: translate centerline ──────────────────────────────────────
    margin_px = float(corridor["margin_px"])
    original_centerline: list[tuple[float, float]] = [
        (float(p["x"]) + translation[0], float(p["y"]) + translation[1])
        for p in corridor["centerline"]
    ]
    aligned_centerline = list(original_centerline)
    original_n = len(aligned_centerline)

    dist0 = math.sqrt(
        (aligned_centerline[0][0] - learner_anchor[0]) ** 2
        + (aligned_centerline[0][1] - learner_anchor[1]) ** 2
    )
    print(f"[corridor_alignment] aligned_centerline[0] → learner_anchor distance: {dist0:.2f} px", flush=True)

    # ── Step 3: learner & corridor Y ranges ───────────────────────────────
    smoothed_pts_raw = smoothed.get("points", [])
    learner_points: list[tuple[float, float]] = [
        (float(p["smoothed_x"]), float(p["smoothed_y"]))
        for p in smoothed_pts_raw
        if p.get("smoothed_x") is not None and p.get("smoothed_y") is not None
    ]
    if not learner_points:
        raise ValueError("learner trajectory_smoothed.json has no valid smoothed points")

    learner_y_min = min(p[1] for p in learner_points)
    learner_y_max = max(p[1] for p in learner_points)
    corridor_y_min = min(p[1] for p in aligned_centerline)
    corridor_y_max = max(p[1] for p in aligned_centerline)
    print(f"[corridor_alignment] Learner Y range: {learner_y_min:.2f} to {learner_y_max:.2f}", flush=True)
    print(f"[corridor_alignment] Corridor Y range (pre-extension): {corridor_y_min:.2f} to {corridor_y_max:.2f}", flush=True)

    # ── Step 4: extend centerline to cover full learner Y range ──────────
    points_added_end = _extend_end(aligned_centerline, learner_y_min)
    points_added_start = _extend_start(aligned_centerline, learner_y_max)

    final_y_min = min(p[1] for p in aligned_centerline)
    final_y_max = max(p[1] for p in aligned_centerline)
    covers = final_y_min <= learner_y_min and final_y_max >= learner_y_max

    print(f"[corridor_alignment] Original corridor centerline points: {original_n}", flush=True)
    print(f"[corridor_alignment] Points added at start: {points_added_start}", flush=True)
    print(f"[corridor_alignment] Points added at end: {points_added_end}", flush=True)
    print(f"[corridor_alignment] Final centerline points: {len(aligned_centerline)}", flush=True)
    print(f"[corridor_alignment] Corridor Y range after extension: {final_y_min:.2f} to {final_y_max:.2f}", flush=True)
    print(f"[corridor_alignment] Corridor covers learner: {covers}", flush=True)

    # ── Step 5: rebuild edges with fixed margin_px ────────────────────────
    left_edge, right_edge, normals, corridor_polygon = _build_corridor_edges(aligned_centerline, margin_px)

    # ── Step 6: per-frame checks (nearest centerline) ─────────────────────
    frame_checks = _compute_nearest_checks(learner_points, smoothed_pts_raw, aligned_centerline, normals, margin_px)

    outside_count = sum(1 for c in frame_checks if c["outside"])
    total = len(frame_checks)
    devs = [c["deviation_px"] for c in frame_checks]
    max_dev = max(devs) if devs else 0.0
    mean_dev = sum(devs) / max(len(devs), 1)
    outside_pct = 100.0 * outside_count / max(total, 1)

    print(f"[corridor_alignment] Outside frames: {outside_count} / {total}", flush=True)
    print(f"[corridor_alignment] Max deviation: {max_dev:.2f} px", flush=True)
    print(f"[corridor_alignment] Mean deviation: {mean_dev:.2f} px", flush=True)
    print(f"[corridor_alignment] Outside percentage: {outside_pct:.1f}%", flush=True)
    if outside_pct > 80.0:
        print(
            "[corridor_alignment] WARNING: learner path mostly outside corridor "
            "— check anchor points or increase margin",
            flush=True,
        )

    # ── Step 7: save aligned_corridor.json ────────────────────────────────
    aligned_json_path = out_dir / ALIGNED_CORRIDOR_FILENAME
    payload: dict[str, Any] = {
        "expert_id": expert_code,
        "learner_run_id": learner_run_id,
        "alignment_mode": "bbox_center_translation_only",
        "margin_px": margin_px,
        "expert_anchor": {"x": round(expert_anchor[0], 2), "y": round(expert_anchor[1], 2)},
        "learner_anchor": {"x": round(learner_anchor[0], 2), "y": round(learner_anchor[1], 2)},
        "translation": {"dx": round(translation[0], 2), "dy": round(translation[1], 2)},
        "original_centerline_points": original_n,
        "extended_centerline_points": len(aligned_centerline),
        "points_added_start": points_added_start,
        "points_added_end": points_added_end,
        "aligned_centerline": [{"x": round(x, 2), "y": round(y, 2)} for x, y in aligned_centerline],
        "left_edge": [{"x": round(x, 2), "y": round(y, 2)} for x, y in left_edge],
        "right_edge": [{"x": round(x, 2), "y": round(y, 2)} for x, y in right_edge],
        "corridor_polygon": [{"x": round(x, 2), "y": round(y, 2)} for x, y in corridor_polygon],
        "frame_checks": frame_checks,
        "summary": {
            "total_frames_checked": total,
            "outside_frames": outside_count,
            "inside_frames": total - outside_count,
            "max_deviation_px": round(max_dev, 2),
            "mean_deviation_px": round(mean_dev, 2),
            "outside_percentage": round(outside_pct, 1),
        },
    }
    _write_json(aligned_json_path, payload)

    # ── Step 8: preview PNG ───────────────────────────────────────────────
    preview_path = out_dir / ALIGNED_CORRIDOR_FRAME0_PREVIEW_FILENAME
    canvas_w, canvas_h = _video_dimensions(learner_video_path)
    _render_translation_preview(
        canvas_w=canvas_w,
        canvas_h=canvas_h,
        aligned_centerline=aligned_centerline,
        left_edge=left_edge,
        right_edge=right_edge,
        corridor_polygon=corridor_polygon,
        learner_points=learner_points,
        frame_checks=frame_checks,
        expert_anchor=expert_anchor,
        learner_anchor=learner_anchor,
        translation=translation,
        points_added_start=points_added_start,
        points_added_end=points_added_end,
        original_n=original_n,
        margin_px=margin_px,
        outside_pct=outside_pct,
        output_path=preview_path,
    )

    # ── Overlay video ─────────────────────────────────────────────────────
    overlay_path = out_dir / ALIGNED_CORRIDOR_OVERLAY_FILENAME
    _render_translation_overlay(
        learner_video_path=learner_video_path,
        aligned_centerline=aligned_centerline,
        corridor_polygon=corridor_polygon,
        left_edge=left_edge,
        right_edge=right_edge,
        learner_points=learner_points,
        frame_checks=frame_checks,
        learner_anchor=learner_anchor,
        output_path=overlay_path,
    )

    print(f"[corridor_alignment] Aligned corridor JSON: {aligned_json_path}", flush=True)
    print(f"[corridor_alignment] Preview image: {preview_path}", flush=True)
    print(f"[corridor_alignment] Overlay video: {overlay_path}", flush=True)

    return {
        "aligned_corridor_json_path": str(aligned_json_path),
        "aligned_corridor_preview_path": str(preview_path),
        "aligned_corridor_preview_url": _storage_url_for(preview_path),
        "aligned_corridor_overlay_video_path": str(overlay_path),
        "aligned_corridor_overlay_video_url": _storage_url_for(overlay_path),
    }


# ── Translation-only helpers ───────────────────────────────────────────────

def _extend_end(
    centerline: list[tuple[float, float]],
    learner_y_min: float,
    max_ext: int = 500,
) -> int:
    """Extend the END of centerline upward (decreasing Y) until it covers learner_y_min."""
    if len(centerline) < 2:
        return 0
    last_pts = centerline[-5:] if len(centerline) >= 5 else centerline[-2:]
    dx = last_pts[-1][0] - last_pts[0][0]
    dy = last_pts[-1][1] - last_pts[0][1]
    total_len = math.sqrt(dx * dx + dy * dy)
    if total_len == 0:
        return 0
    sx, sy = dx / total_len, dy / total_len

    steps = [
        math.sqrt(
            (last_pts[i][0] - last_pts[i - 1][0]) ** 2
            + (last_pts[i][1] - last_pts[i - 1][1]) ** 2
        )
        for i in range(1, len(last_pts))
    ]
    avg_step = sum(steps) / len(steps) if steps else 1.0

    count = 0
    while centerline[-1][1] > learner_y_min and count < max_ext:
        lx, ly = centerline[-1]
        centerline.append((lx + sx * avg_step, ly + sy * avg_step))
        count += 1
    return count


def _extend_start(
    centerline: list[tuple[float, float]],
    learner_y_max: float,
    max_ext: int = 500,
) -> int:
    """Prepend to the START of centerline downward (increasing Y) until it covers learner_y_max."""
    if len(centerline) < 2:
        return 0
    first_pts = centerline[:5] if len(centerline) >= 5 else centerline[:2]
    dx = first_pts[-1][0] - first_pts[0][0]
    dy = first_pts[-1][1] - first_pts[0][1]
    total_len = math.sqrt(dx * dx + dy * dy)
    if total_len == 0:
        return 0
    sx, sy = dx / total_len, dy / total_len

    steps = [
        math.sqrt(
            (first_pts[i][0] - first_pts[i - 1][0]) ** 2
            + (first_pts[i][1] - first_pts[i - 1][1]) ** 2
        )
        for i in range(1, len(first_pts))
    ]
    avg_step = sum(steps) / len(steps) if steps else 1.0

    # Build prepend list then splice in once — avoids O(n²) insert-at-0
    prepend: list[tuple[float, float]] = []
    cur = centerline[0]
    count = 0
    while cur[1] < learner_y_max and count < max_ext:
        new_pt = (cur[0] - sx * avg_step, cur[1] - sy * avg_step)
        prepend.append(new_pt)
        cur = new_pt
        count += 1
    prepend.reverse()
    centerline[:0] = prepend
    return count


def _compute_nearest_checks(
    learner_points: list[tuple[float, float]],
    smoothed_pts_raw: list[dict[str, Any]],
    aligned_centerline: list[tuple[float, float]],
    normals: list[tuple[float, float]],
    margin_px: float,
) -> list[dict[str, Any]]:
    checks: list[dict[str, Any]] = []
    for i, (lx, ly) in enumerate(learner_points):
        min_dist = float("inf")
        nearest_i = 0
        for ci, (cx, cy) in enumerate(aligned_centerline):
            d = math.sqrt((lx - cx) ** 2 + (ly - cy) ** 2)
            if d < min_dist:
                min_dist = d
                nearest_i = ci
        nx, ny = normals[nearest_i]
        cx, cy = aligned_centerline[nearest_i]
        signed_dev = (lx - cx) * nx + (ly - cy) * ny
        deviation_px = abs(signed_dev)
        frame_index = smoothed_pts_raw[i].get("frame_index", i) if i < len(smoothed_pts_raw) else i
        checks.append(
            {
                "frame_index": frame_index,
                "learner_x": round(lx, 2),
                "learner_y": round(ly, 2),
                "nearest_center_x": round(cx, 2),
                "nearest_center_y": round(cy, 2),
                "deviation_px": round(deviation_px, 2),
                "signed_deviation_px": round(signed_dev, 2),
                "outside": bool(deviation_px > margin_px),
            }
        )
    return checks


# BGR constants used only for translation renderer
_COLOR_EXT_EDGE = (0, 100, 0)        # dark green for extrapolated edges
_COLOR_EXT_CL = (80, 80, 80)         # grey for extrapolated centerline
_COLOR_EXPERT_ANCHOR = (255, 255, 255)  # white ring
_COLOR_LEARNER_ANCHOR_PT = (255, 50, 50)  # blue-ish dot


def _draw_translation_corridor(
    frame: np.ndarray,
    aligned_centerline: list[tuple[float, float]],
    corridor_polygon: list[tuple[float, float]],
    left_edge: list[tuple[float, float]],
    right_edge: list[tuple[float, float]],
    learner_points: list[tuple[float, float]],
    frame_checks: list[dict[str, Any]],
    learner_anchor: tuple[float, float],
) -> None:
    poly_pts = _pts_i32(corridor_polygon)
    overlay = frame.copy()
    cv2.fillPoly(overlay, [poly_pts], _COLOR_CORRIDOR_FILL)
    cv2.addWeighted(overlay, _CORRIDOR_ALPHA, frame, 1 - _CORRIDOR_ALPHA, 0, dst=frame)

    if len(left_edge) >= 2:
        cv2.polylines(frame, [_pts_i32(left_edge)], False, _COLOR_EDGE, 1)
    if len(right_edge) >= 2:
        cv2.polylines(frame, [_pts_i32(right_edge)], False, _COLOR_EDGE, 1)
    if len(aligned_centerline) >= 2:
        cv2.polylines(frame, [_pts_i32(aligned_centerline)], False, _COLOR_CENTERLINE, 2)
    if len(learner_points) >= 2:
        cv2.polylines(frame, [_pts_i32(learner_points)], False, _COLOR_LEARNER, 2)

    for chk in frame_checks:
        pt = (int(round(chk["learner_x"])), int(round(chk["learner_y"])))
        if chk["outside"]:
            cv2.circle(frame, pt, 6, _COLOR_OUTSIDE_PT, -1)
        else:
            cv2.circle(frame, pt, 3, _COLOR_INSIDE_PT, -1)

    la_pt = (int(round(learner_anchor[0])), int(round(learner_anchor[1])))
    cv2.circle(frame, la_pt, 10, _COLOR_EXPERT_ANCHOR, 2)
    cv2.circle(frame, la_pt, 5, _COLOR_LEARNER_ANCHOR_PT, -1)


def _render_translation_preview(
    *,
    canvas_w: int,
    canvas_h: int,
    aligned_centerline: list[tuple[float, float]],
    left_edge: list[tuple[float, float]],
    right_edge: list[tuple[float, float]],
    corridor_polygon: list[tuple[float, float]],
    learner_points: list[tuple[float, float]],
    frame_checks: list[dict[str, Any]],
    expert_anchor: tuple[float, float],
    learner_anchor: tuple[float, float],
    translation: tuple[float, float],
    points_added_start: int,
    points_added_end: int,
    original_n: int,
    margin_px: float,
    outside_pct: float,
    output_path: Path,
) -> None:
    canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)

    # Semi-transparent corridor fill
    poly_pts = _pts_i32(corridor_polygon)
    ov = canvas.copy()
    cv2.fillPoly(ov, [poly_pts], _COLOR_CORRIDOR_FILL)
    cv2.addWeighted(ov, _CORRIDOR_ALPHA, canvas, 1 - _CORRIDOR_ALPHA, 0, dst=canvas)

    # Index boundaries for original section
    orig_start = points_added_start
    orig_end = points_added_start + original_n  # exclusive end

    # Helper: draw a segment of left/right edge with a given colour
    def _seg(edge: list[tuple[float, float]], a: int, b: int, colour: tuple[int, int, int], thickness: int = 1) -> None:
        seg = edge[a:b]
        if len(seg) >= 2:
            cv2.polylines(canvas, [_pts_i32(seg)], False, colour, thickness)

    # Extended start edges (dark green)
    if points_added_start > 0:
        _seg(left_edge, 0, orig_start + 1, _COLOR_EXT_EDGE)
        _seg(right_edge, 0, orig_start + 1, _COLOR_EXT_EDGE)
    # Original edges (yellow)
    _seg(left_edge, orig_start, orig_end, _COLOR_EDGE)
    _seg(right_edge, orig_start, orig_end, _COLOR_EDGE)
    # Extended end edges (dark green)
    if points_added_end > 0:
        _seg(left_edge, orig_end - 1, len(left_edge), _COLOR_EXT_EDGE)
        _seg(right_edge, orig_end - 1, len(right_edge), _COLOR_EXT_EDGE)

    # Centerline segments
    if points_added_start > 0 and orig_start + 1 <= len(aligned_centerline):
        seg = aligned_centerline[: orig_start + 1]
        if len(seg) >= 2:
            cv2.polylines(canvas, [_pts_i32(seg)], False, _COLOR_EXT_CL, 1)
    orig_cl = aligned_centerline[orig_start:orig_end]
    if len(orig_cl) >= 2:
        cv2.polylines(canvas, [_pts_i32(orig_cl)], False, _COLOR_CENTERLINE, 2)
    if points_added_end > 0 and orig_end <= len(aligned_centerline):
        seg = aligned_centerline[orig_end - 1:]
        if len(seg) >= 2:
            cv2.polylines(canvas, [_pts_i32(seg)], False, _COLOR_EXT_CL, 1)

    # Learner path
    if len(learner_points) >= 2:
        cv2.polylines(canvas, [_pts_i32(learner_points)], False, _COLOR_LEARNER, 2)

    # Per-frame deviation dots
    for chk in frame_checks:
        pt = (int(round(chk["learner_x"])), int(round(chk["learner_y"])))
        if chk["outside"]:
            cv2.circle(canvas, pt, 6, _COLOR_OUTSIDE_PT, -1)
        else:
            cv2.circle(canvas, pt, 3, _COLOR_INSIDE_PT, -1)

    # Anchors — expert and learner coincide in aligned space (by definition of translation)
    # Draw as concentric rings: white outer = expert, blue inner = learner
    la_pt = (int(round(learner_anchor[0])), int(round(learner_anchor[1])))
    cv2.circle(canvas, la_pt, 12, _COLOR_EXPERT_ANCHOR, 2)
    cv2.circle(canvas, la_pt, 6, _COLOR_LEARNER_START, -1)
    cv2.putText(
        canvas, "EXPERT ANCHOR",
        (la_pt[0] + 15, la_pt[1] - 8),
        cv2.FONT_HERSHEY_SIMPLEX, 0.55, _COLOR_EXPERT_ANCHOR, 2, cv2.LINE_AA,
    )
    cv2.putText(
        canvas, "LEARNER ANCHOR",
        (la_pt[0] + 15, la_pt[1] + 16),
        cv2.FONT_HERSHEY_SIMPLEX, 0.55, _COLOR_LEARNER_START, 2, cv2.LINE_AA,
    )

    # Info text block in top-left
    dx, dy = translation
    info_lines = [
        f"scale=1.00 (translation only)",
        f"dx={dx:.1f}  dy={dy:.1f}  margin={int(margin_px)}px",
        f"outside={outside_pct:.1f}%",
        f"orig_cl={original_n}  +start={points_added_start}  +end={points_added_end}",
    ]
    for li, txt in enumerate(info_lines):
        cv2.putText(
            canvas, txt,
            (18, 30 + li * 26),
            cv2.FONT_HERSHEY_SIMPLEX, 0.65, _COLOR_CENTERLINE, 2, cv2.LINE_AA,
        )

    cv2.imwrite(str(output_path), canvas)


def _render_translation_overlay(
    *,
    learner_video_path: str | None,
    aligned_centerline: list[tuple[float, float]],
    corridor_polygon: list[tuple[float, float]],
    left_edge: list[tuple[float, float]],
    right_edge: list[tuple[float, float]],
    learner_points: list[tuple[float, float]],
    frame_checks: list[dict[str, Any]],
    learner_anchor: tuple[float, float],
    output_path: Path,
) -> None:
    cap: cv2.VideoCapture | None = None
    fps = 30.0
    w, h = 1920, 1080

    if learner_video_path:
        cap = cv2.VideoCapture(str(learner_video_path))
        if not cap.isOpened():
            cap = None

    if cap is not None:
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (w, h))

    def _draw(frame: np.ndarray) -> None:
        _draw_translation_corridor(
            frame, aligned_centerline, corridor_polygon,
            left_edge, right_edge, learner_points, frame_checks, learner_anchor,
        )

    try:
        if cap is not None:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                _draw(frame)
                writer.write(frame)
            cap.release()
        else:
            blank = np.zeros((h, w, 3), dtype=np.uint8)
            for _ in range(max(len(aligned_centerline), 30)):
                frame = blank.copy()
                _draw(frame)
                writer.write(frame)
    finally:
        writer.release()


# ── Archive: frame-0 bbox anchor + scale (kept, not used by default) ─────────

def align_corridor_to_learner_frame0_anchor(
    *,
    expert_raw_path: str,
    corridor_path: str,
    learner_raw_path: str,
    learner_smoothed_path: str,
    learner_run_id: str,
    output_dir: str,
    expert_code: str,
    learner_video_path: str | None = None,
) -> dict[str, Any]:
    """Align expert corridor to learner using frame-0 mask bbox anchor and scale."""
    expert_raw_p = Path(expert_raw_path).expanduser().resolve()
    corridor_p = Path(corridor_path).expanduser().resolve()
    learner_raw_p = Path(learner_raw_path).expanduser().resolve()
    smoothed_p = Path(learner_smoothed_path).expanduser().resolve()
    out_dir = Path(output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    expert_raw = json.loads(expert_raw_p.read_text(encoding="utf-8"))
    corridor = json.loads(corridor_p.read_text(encoding="utf-8"))
    learner_raw = json.loads(learner_raw_p.read_text(encoding="utf-8"))
    smoothed = json.loads(smoothed_p.read_text(encoding="utf-8"))

    expert_frame0 = _frame0_payload(expert_raw, "expert")
    learner_frame0 = _frame0_payload(learner_raw, "learner")
    expert_anchor = _xy_tuple(expert_frame0.get("bbox_center"), "expert frame 0 bbox_center")
    learner_anchor = _xy_tuple(learner_frame0.get("bbox_center"), "learner frame 0 bbox_center")
    expert_bbox = _bbox_tuple(expert_frame0.get("mask_bbox"), "expert frame 0 mask_bbox")
    learner_bbox = _bbox_tuple(learner_frame0.get("mask_bbox"), "learner frame 0 mask_bbox")

    expert_bbox_height = expert_bbox[3] - expert_bbox[1]
    learner_bbox_height = learner_bbox[3] - learner_bbox[1]
    if expert_bbox_height <= 0:
        raise ValueError(f"Expert frame 0 mask_bbox has invalid height: {expert_bbox_height}")
    if learner_bbox_height <= 0:
        raise ValueError(f"Learner frame 0 mask_bbox has invalid height: {learner_bbox_height}")

    translation = (
        learner_anchor[0] - expert_anchor[0],
        learner_anchor[1] - expert_anchor[1],
    )
    scale = learner_bbox_height / expert_bbox_height

    aligned_centerline = _transform_point_list(corridor["centerline"], expert_anchor, translation, scale)
    left_edge = _transform_point_list(corridor["left_edge"], expert_anchor, translation, scale)
    right_edge = _transform_point_list(corridor["right_edge"], expert_anchor, translation, scale)
    corridor_polygon = _transform_point_list(corridor["corridor_polygon"], expert_anchor, translation, scale)
    learner_points = [
        (float(p["smoothed_x"]), float(p["smoothed_y"]))
        for p in smoothed.get("points", [])
        if p.get("smoothed_x") is not None and p.get("smoothed_y") is not None
    ]

    print(f"[corridor_alignment] Expert bbox height: {expert_bbox_height:.2f} px", flush=True)
    print(f"[corridor_alignment] Learner bbox height: {learner_bbox_height:.2f} px", flush=True)
    print(f"[corridor_alignment] Scale: {scale:.4f}", flush=True)
    print(
        f"[corridor_alignment] Translation: dx={translation[0]:.2f}, dy={translation[1]:.2f}",
        flush=True,
    )

    aligned_corridor_path = out_dir / ALIGNED_CORRIDOR_FILENAME
    preview_path = out_dir / ALIGNED_CORRIDOR_FRAME0_PREVIEW_FILENAME
    overlay_path = out_dir / ALIGNED_CORRIDOR_OVERLAY_FILENAME

    payload: dict[str, Any] = {
        "expert_code": expert_code,
        "learner_run_id": learner_run_id,
        "alignment_mode": "frame0_bbox_anchor_scale",
        "expert_anchor": {"x": round(expert_anchor[0], 2), "y": round(expert_anchor[1], 2)},
        "learner_anchor": {"x": round(learner_anchor[0], 2), "y": round(learner_anchor[1], 2)},
        "translation": {"dx": round(translation[0], 2), "dy": round(translation[1], 2)},
        "scale": round(scale, 6),
        "expert_bbox_height": round(expert_bbox_height, 2),
        "learner_bbox_height": round(learner_bbox_height, 2),
        "centerline": aligned_centerline,
        "left_edge": left_edge,
        "right_edge": right_edge,
        "corridor_polygon": corridor_polygon,
    }
    _write_json(aligned_corridor_path, payload)

    aligned_centerline_pts = _dict_points_to_tuples(aligned_centerline)
    left_edge_pts = _dict_points_to_tuples(left_edge)
    right_edge_pts = _dict_points_to_tuples(right_edge)
    corridor_polygon_pts = _dict_points_to_tuples(corridor_polygon)
    annotation = f"scale={scale:.3f}  translation=({translation[0]:.1f}, {translation[1]:.1f})"

    canvas_w, canvas_h = _video_dimensions(learner_video_path)
    _render_frame0_preview(
        canvas_w=canvas_w,
        canvas_h=canvas_h,
        aligned_centerline=aligned_centerline_pts,
        left_edge=left_edge_pts,
        right_edge=right_edge_pts,
        corridor_polygon=corridor_polygon_pts,
        learner_points=learner_points,
        expert_anchor_aligned=learner_anchor,
        learner_anchor=learner_anchor,
        annotation=annotation,
        output_path=preview_path,
    )
    _render_frame0_overlay_video(
        learner_video_path=learner_video_path,
        aligned_centerline=aligned_centerline_pts,
        corridor_polygon=corridor_polygon_pts,
        left_edge=left_edge_pts,
        right_edge=right_edge_pts,
        learner_points=learner_points,
        expert_anchor_aligned=learner_anchor,
        learner_anchor=learner_anchor,
        output_path=overlay_path,
    )

    print(f"[corridor_alignment] Aligned corridor path: {aligned_corridor_path}", flush=True)
    print(f"[corridor_alignment] Preview image path: {preview_path}", flush=True)
    print(f"[corridor_alignment] Overlay video path: {overlay_path}", flush=True)

    return {
        "aligned_corridor_json_path": str(aligned_corridor_path),
        "aligned_corridor_preview_path": str(preview_path),
        "aligned_corridor_preview_url": _storage_url_for(preview_path),
        "aligned_corridor_overlay_video_path": str(overlay_path),
        "aligned_corridor_overlay_video_url": _storage_url_for(overlay_path),
    }


def align_corridor_to_learner(
    corridor_path: str,
    learner_smoothed_path: str,
    learner_run_id: str,
    output_dir: str,
    expert_code: str,
    learner_video_path: str | None = None,
    resample_n: int = 100,
) -> dict[str, Any]:
    """Align expert corridor to learner trajectory using arc-length progress.

    Parameters
    ----------
    corridor_path:
        Absolute path to expert corridor.json.
    learner_smoothed_path:
        Absolute path to learner trajectory_smoothed.json.
    learner_run_id:
        UUID string for the learner run.
    output_dir:
        Directory to write aligned_corridor_progress.json, preview PNG and
        overlay MP4.
    expert_code:
        Expert identifier string (e.g. "straight_line_v1").
    learner_video_path:
        Optional path to the original learner video; used for overlay and
        canvas size.
    resample_n:
        Number of equi-spaced arc-length samples (default 100).

    Returns
    -------
    dict with output paths and public storage URLs.
    """
    corridor_p = Path(corridor_path).expanduser().resolve()
    smoothed_p = Path(learner_smoothed_path).expanduser().resolve()
    out_dir = Path(output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    corridor = json.loads(corridor_p.read_text(encoding="utf-8"))
    smoothed = json.loads(smoothed_p.read_text(encoding="utf-8"))

    # ── Step 1: Extract points ─────────────────────────────────────────────
    expert_centerline: list[tuple[float, float]] = [
        (p["x"], p["y"]) for p in corridor["centerline"]
    ]
    learner_points: list[tuple[float, float]] = [
        (p["smoothed_x"], p["smoothed_y"]) for p in smoothed["points"]
    ]
    margin_px: float = float(corridor["margin_px"])

    if len(expert_centerline) < 3:
        raise ValueError(
            f"Expert corridor has too few centerline points: {len(expert_centerline)}"
        )
    if len(learner_points) < 3:
        raise ValueError(
            f"Learner trajectory has too few smoothed points: {len(learner_points)}"
        )

    # ── Step 2: Resample by arc-length ────────────────────────────────────
    expert_resampled = _resample_by_arclength(expert_centerline, n=resample_n)
    learner_resampled = _resample_by_arclength(learner_points, n=resample_n)

    expert_arc_length = _arc_length(expert_centerline)
    learner_arc_length = _arc_length(learner_points)
    arc_length_ratio = learner_arc_length / max(expert_arc_length, 1e-6)

    print(f"[corridor_alignment] Expert arc-length: {expert_arc_length:.1f} px", flush=True)
    print(f"[corridor_alignment] Learner arc-length: {learner_arc_length:.1f} px", flush=True)
    print(f"[corridor_alignment] Arc-length ratio: {arc_length_ratio:.3f}", flush=True)
    if arc_length_ratio > 2.0 or arc_length_ratio < 0.5:
        print(
            f"[corridor_alignment] WARNING: arc-length ratio {arc_length_ratio:.3f} is "
            "outside [0.5, 2.0] — paths may differ significantly",
            flush=True,
        )

    # ── Step 3: Compute directions ────────────────────────────────────────
    expert_direction = _compute_direction(expert_resampled, n=10)
    learner_direction = _compute_direction(learner_resampled, n=10)
    rotation_angle = learner_direction - expert_direction
    rotation_angle_deg = math.degrees(rotation_angle)

    print(f"[corridor_alignment] Rotation angle: {rotation_angle_deg:.2f} degrees", flush=True)
    if abs(rotation_angle_deg) > 45:
        print(
            f"[corridor_alignment] WARNING: large rotation ({rotation_angle_deg:.2f}°) "
            "— retrying direction with first 20 points",
            flush=True,
        )
        expert_direction = _compute_direction(expert_resampled, n=20)
        learner_direction = _compute_direction(learner_resampled, n=20)
        rotation_angle = learner_direction - expert_direction
        rotation_angle_deg = math.degrees(rotation_angle)
        print(
            f"[corridor_alignment] Rotation angle (20-pt): {rotation_angle_deg:.2f} degrees",
            flush=True,
        )
        if abs(rotation_angle_deg) > 45:
            print(
                "[corridor_alignment] WARNING: large rotation still detected — "
                "verify first points are clean",
                flush=True,
            )

    # ── Step 4: Apply alignment transform ────────────────────────────────
    expert_start = expert_resampled[0]
    learner_start = learner_resampled[0]

    aligned_centerline: list[tuple[float, float]] = []
    for x, y in expert_resampled:
        px = x - expert_start[0]
        py = y - expert_start[1]
        rx, ry = _rotate_point(px, py, rotation_angle)
        aligned_centerline.append((rx + learner_start[0], ry + learner_start[1]))

    dx_check = abs(aligned_centerline[0][0] - learner_start[0])
    dy_check = abs(aligned_centerline[0][1] - learner_start[1])
    start_ok = dx_check < 0.01 and dy_check < 0.01
    print(
        f"[corridor_alignment] aligned_centerline[0] == learner_start: {start_ok}",
        flush=True,
    )
    if not start_ok:
        print(
            f"[corridor_alignment] WARNING: alignment start mismatch "
            f"({dx_check:.4f}, {dy_check:.4f}) px",
            flush=True,
        )

    # ── Step 5: Build corridor edges ──────────────────────────────────────
    left_edge, right_edge, normals, corridor_polygon = _build_corridor_edges(
        aligned_centerline, margin_px
    )

    # ── Step 6: Progress checks ───────────────────────────────────────────
    progress_checks = _compute_progress_checks(
        learner_resampled, aligned_centerline, normals, margin_px, resample_n
    )

    outside_count = sum(1 for c in progress_checks if c["outside"])
    deviations = [c["deviation_px"] for c in progress_checks]
    max_dev = max(deviations) if deviations else 0.0
    mean_dev = sum(deviations) / max(len(deviations), 1)
    translation = (
        learner_start[0] - expert_start[0],
        learner_start[1] - expert_start[1],
    )

    print(
        f"[corridor_alignment] Translation: ({translation[0]:.2f}, {translation[1]:.2f})",
        flush=True,
    )
    print(
        f"[corridor_alignment] Outside progress points: {outside_count} / {resample_n}",
        flush=True,
    )
    print(f"[corridor_alignment] Max deviation: {max_dev:.2f} px", flush=True)
    print(f"[corridor_alignment] Mean deviation: {mean_dev:.2f} px", flush=True)
    if outside_count > 80:
        print(
            "[corridor_alignment] WARNING: learner path mostly outside corridor "
            "— check alignment or increase margin",
            flush=True,
        )

    # ── Step 7: Save aligned_corridor_progress.json ───────────────────────
    progress_json_path = out_dir / ALIGNED_CORRIDOR_PROGRESS_FILENAME
    progress_payload: dict[str, Any] = {
        "expert_code": expert_code,
        "learner_run_id": learner_run_id,
        "alignment_mode": "arc_length_progress",
        "resample_n": resample_n,
        "margin_px": margin_px,
        "expert_arc_length_px": round(expert_arc_length, 2),
        "learner_arc_length_px": round(learner_arc_length, 2),
        "arc_length_ratio": round(arc_length_ratio, 4),
        "expert_start": {"x": round(expert_start[0], 2), "y": round(expert_start[1], 2)},
        "learner_start": {"x": round(learner_start[0], 2), "y": round(learner_start[1], 2)},
        "translation": {"dx": round(translation[0], 2), "dy": round(translation[1], 2)},
        "rotation_angle_deg": round(rotation_angle_deg, 4),
        "aligned_centerline": [
            {"x": round(x, 2), "y": round(y, 2)} for x, y in aligned_centerline
        ],
        "left_edge": [{"x": round(x, 2), "y": round(y, 2)} for x, y in left_edge],
        "right_edge": [{"x": round(x, 2), "y": round(y, 2)} for x, y in right_edge],
        "corridor_polygon": [
            {"x": round(x, 2), "y": round(y, 2)} for x, y in corridor_polygon
        ],
        "learner_resampled": [
            {"x": round(x, 2), "y": round(y, 2)} for x, y in learner_resampled
        ],
        "progress_checks": progress_checks,
    }
    _write_json(progress_json_path, progress_payload)

    # ── Step 8: Preview PNG ───────────────────────────────────────────────
    preview_path = out_dir / ALIGNED_CORRIDOR_PREVIEW_FILENAME
    canvas_w, canvas_h = _video_dimensions(learner_video_path)
    _render_preview(
        canvas_w=canvas_w,
        canvas_h=canvas_h,
        aligned_centerline=aligned_centerline,
        left_edge=left_edge,
        right_edge=right_edge,
        corridor_polygon=corridor_polygon,
        learner_resampled=learner_resampled,
        progress_checks=progress_checks,
        expert_start=expert_start,
        learner_start=learner_start,
        output_path=preview_path,
    )
    print(f"[corridor_alignment] Preview image path: {preview_path}", flush=True)

    # ── Step 9: Learner video overlay ─────────────────────────────────────
    overlay_path = out_dir / ALIGNED_CORRIDOR_OVERLAY_FILENAME
    _render_overlay_video(
        learner_video_path=learner_video_path,
        aligned_centerline=aligned_centerline,
        corridor_polygon=corridor_polygon,
        left_edge=left_edge,
        right_edge=right_edge,
        learner_full_points=learner_points,
        progress_checks=progress_checks,
        output_path=overlay_path,
    )
    print(f"[corridor_alignment] Overlay video path: {overlay_path}", flush=True)

    return {
        "aligned_corridor_progress_json_path": str(progress_json_path),
        "aligned_corridor_progress_preview_path": str(preview_path),
        "aligned_corridor_progress_preview_url": _storage_url_for(preview_path),
        "aligned_corridor_overlay_video_path": str(overlay_path),
        "aligned_corridor_overlay_video_url": _storage_url_for(overlay_path),
    }


# ──────────────────────────────────────────────────────────────────────────────
# Math helpers
# ──────────────────────────────────────────────────────────────────────────────

def _frame0_payload(raw: dict[str, Any], label: str) -> dict[str, Any]:
    frames = raw.get("frames")
    if not isinstance(frames, list) or not frames:
        raise ValueError(f"{label} raw.json has no frames")
    frame0 = frames[0]
    if not isinstance(frame0, dict):
        raise ValueError(f"{label} raw.json frames[0] is invalid")
    return frame0


def _xy_tuple(value: Any, label: str) -> tuple[float, float]:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError(f"{label} must be a 2-value point")
    return float(value[0]), float(value[1])


def _bbox_tuple(value: Any, label: str) -> tuple[float, float, float, float]:
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        raise ValueError(f"{label} must be a 4-value bbox")
    return float(value[0]), float(value[1]), float(value[2]), float(value[3])


def _transform_point_list(
    points: list[dict[str, Any]],
    expert_anchor: tuple[float, float],
    translation: tuple[float, float],
    scale: float,
) -> list[dict[str, Any]]:
    return [
        _transform_point_dict(point, expert_anchor, translation, scale)
        for point in points
    ]


def _transform_point_dict(
    point: dict[str, Any],
    expert_anchor: tuple[float, float],
    translation: tuple[float, float],
    scale: float,
) -> dict[str, Any]:
    x = float(point["x"])
    y = float(point["y"])
    scaled_x = expert_anchor[0] + scale * (x - expert_anchor[0])
    scaled_y = expert_anchor[1] + scale * (y - expert_anchor[1])
    transformed = {
        key: value
        for key, value in point.items()
        if key not in {"x", "y"}
    }
    transformed["x"] = round(scaled_x + translation[0], 2)
    transformed["y"] = round(scaled_y + translation[1], 2)
    return transformed


def _dict_points_to_tuples(points: list[dict[str, Any]]) -> list[tuple[float, float]]:
    return [(float(point["x"]), float(point["y"])) for point in points]


def _arc_length(points: list[tuple[float, float]]) -> float:
    total = 0.0
    for i in range(1, len(points)):
        dx = points[i][0] - points[i - 1][0]
        dy = points[i][1] - points[i - 1][1]
        total += math.sqrt(dx * dx + dy * dy)
    return total


def _resample_by_arclength(
    points: list[tuple[float, float]], n: int = 100
) -> list[tuple[float, float]]:
    distances: list[float] = [0.0]
    for i in range(1, len(points)):
        dx = points[i][0] - points[i - 1][0]
        dy = points[i][1] - points[i - 1][1]
        distances.append(distances[-1] + math.sqrt(dx * dx + dy * dy))

    total_length = distances[-1]
    targets = [total_length * i / (n - 1) for i in range(n)]

    resampled: list[tuple[float, float]] = []
    for t in targets:
        placed = False
        for i in range(1, len(distances)):
            if distances[i] >= t:
                seg_len = distances[i] - distances[i - 1]
                alpha = (t - distances[i - 1]) / seg_len if seg_len > 0 else 0.0
                x = points[i - 1][0] + alpha * (points[i][0] - points[i - 1][0])
                y = points[i - 1][1] + alpha * (points[i][1] - points[i - 1][1])
                resampled.append((x, y))
                placed = True
                break
        if not placed:
            resampled.append(points[-1])

    return resampled


def _compute_direction(
    resampled_points: list[tuple[float, float]], n: int = 10
) -> float:
    dx = resampled_points[n - 1][0] - resampled_points[0][0]
    dy = resampled_points[n - 1][1] - resampled_points[0][1]
    return math.atan2(dy, dx)


def _rotate_point(px: float, py: float, angle: float) -> tuple[float, float]:
    cos_a = math.cos(angle)
    sin_a = math.sin(angle)
    return (px * cos_a - py * sin_a, px * sin_a + py * cos_a)


def _build_corridor_edges(
    aligned_centerline: list[tuple[float, float]],
    margin_px: float,
) -> tuple[
    list[tuple[float, float]],
    list[tuple[float, float]],
    list[tuple[float, float]],
    list[tuple[float, float]],
]:
    left_edge: list[tuple[float, float]] = []
    right_edge: list[tuple[float, float]] = []
    normals: list[tuple[float, float]] = []
    n = len(aligned_centerline)

    for i in range(n):
        if i == 0:
            dx = aligned_centerline[1][0] - aligned_centerline[0][0]
            dy = aligned_centerline[1][1] - aligned_centerline[0][1]
        elif i == n - 1:
            dx = aligned_centerline[-1][0] - aligned_centerline[-2][0]
            dy = aligned_centerline[-1][1] - aligned_centerline[-2][1]
        else:
            dx = aligned_centerline[i + 1][0] - aligned_centerline[i - 1][0]
            dy = aligned_centerline[i + 1][1] - aligned_centerline[i - 1][1]

        length = math.sqrt(dx * dx + dy * dy)
        if length == 0:
            length = 1.0
        dx /= length
        dy /= length

        nx, ny = -dy, dx

        if i > 0:
            prev_nx, prev_ny = normals[-1]
            if (nx * prev_nx + ny * prev_ny) < 0:
                nx, ny = -nx, -ny

        normals.append((nx, ny))
        cx, cy = aligned_centerline[i]
        left_edge.append((cx + margin_px * nx, cy + margin_px * ny))
        right_edge.append((cx - margin_px * nx, cy - margin_px * ny))

    corridor_polygon = left_edge + list(reversed(right_edge)) + [left_edge[0]]
    return left_edge, right_edge, normals, corridor_polygon


def _compute_progress_checks(
    learner_resampled: list[tuple[float, float]],
    aligned_centerline: list[tuple[float, float]],
    normals: list[tuple[float, float]],
    margin_px: float,
    resample_n: int,
) -> list[dict[str, Any]]:
    checks: list[dict[str, Any]] = []
    for i in range(resample_n):
        lx, ly = learner_resampled[i]
        cx, cy = aligned_centerline[i]
        nx, ny = normals[i]
        deviation = (lx - cx) * nx + (ly - cy) * ny
        deviation_px = abs(deviation)
        checks.append({
            "progress_index": i,
            "progress_ratio": round(i / (resample_n - 1) if resample_n > 1 else 0.0, 4),
            "learner_x": round(lx, 2),
            "learner_y": round(ly, 2),
            "center_x": round(cx, 2),
            "center_y": round(cy, 2),
            "deviation_px": round(deviation_px, 2),
            "signed_deviation_px": round(deviation, 2),
            "outside": bool(deviation_px > margin_px),
        })
    return checks


# ──────────────────────────────────────────────────────────────────────────────
# Rendering helpers
# ──────────────────────────────────────────────────────────────────────────────

def _pts_i32(pts: list[tuple[float, float]]) -> np.ndarray:
    return np.array(
        [[int(round(x)), int(round(y))] for x, y in pts], dtype=np.int32
    )


def _draw_aligned_corridor_overlay(
    frame: np.ndarray,
    aligned_centerline: list[tuple[float, float]],
    corridor_polygon: list[tuple[float, float]],
    left_edge: list[tuple[float, float]],
    right_edge: list[tuple[float, float]],
    learner_full_pts: list[tuple[float, float]],
    progress_checks: list[dict[str, Any]],
) -> None:
    """Draw aligned corridor overlay onto an existing BGR frame in-place."""
    poly_pts = _pts_i32(corridor_polygon)
    overlay = frame.copy()
    cv2.fillPoly(overlay, [poly_pts], _COLOR_CORRIDOR_FILL)
    cv2.addWeighted(overlay, _CORRIDOR_ALPHA, frame, 1 - _CORRIDOR_ALPHA, 0, dst=frame)

    if len(left_edge) >= 2:
        cv2.polylines(frame, [_pts_i32(left_edge)], False, _COLOR_EDGE, 1)
    if len(right_edge) >= 2:
        cv2.polylines(frame, [_pts_i32(right_edge)], False, _COLOR_EDGE, 1)
    if len(aligned_centerline) >= 2:
        cv2.polylines(frame, [_pts_i32(aligned_centerline)], False, _COLOR_CENTERLINE, 2)

    if len(learner_full_pts) >= 2:
        cv2.polylines(frame, [_pts_i32(learner_full_pts)], False, _COLOR_LEARNER, 2)

    for chk in progress_checks:
        pt = (int(round(chk["learner_x"])), int(round(chk["learner_y"])))
        if chk["outside"]:
            cv2.circle(frame, pt, 6, _COLOR_OUTSIDE_PT, -1)
        else:
            cv2.circle(frame, pt, 3, _COLOR_INSIDE_PT, -1)


def _render_preview(
    *,
    canvas_w: int,
    canvas_h: int,
    aligned_centerline: list[tuple[float, float]],
    left_edge: list[tuple[float, float]],
    right_edge: list[tuple[float, float]],
    corridor_polygon: list[tuple[float, float]],
    learner_resampled: list[tuple[float, float]],
    progress_checks: list[dict[str, Any]],
    expert_start: tuple[float, float],
    learner_start: tuple[float, float],
    output_path: Path,
) -> None:
    canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)

    # Filled corridor polygon
    poly_pts = _pts_i32(corridor_polygon)
    overlay = canvas.copy()
    cv2.fillPoly(overlay, [poly_pts], _COLOR_CORRIDOR_FILL)
    cv2.addWeighted(overlay, _CORRIDOR_ALPHA, canvas, 1 - _CORRIDOR_ALPHA, 0, dst=canvas)

    # Edges and centerline
    if len(left_edge) >= 2:
        cv2.polylines(canvas, [_pts_i32(left_edge)], False, _COLOR_EDGE, 1)
    if len(right_edge) >= 2:
        cv2.polylines(canvas, [_pts_i32(right_edge)], False, _COLOR_EDGE, 1)
    if len(aligned_centerline) >= 2:
        cv2.polylines(canvas, [_pts_i32(aligned_centerline)], False, _COLOR_CENTERLINE, 2)

    # Learner resampled path
    if len(learner_resampled) >= 2:
        cv2.polylines(canvas, [_pts_i32(learner_resampled)], False, _COLOR_LEARNER, 2)

    # Progress point dots
    for chk in progress_checks:
        pt = (int(round(chk["learner_x"])), int(round(chk["learner_y"])))
        if chk["outside"]:
            cv2.circle(canvas, pt, 6, _COLOR_OUTSIDE_PT, -1)
        else:
            cv2.circle(canvas, pt, 3, _COLOR_INSIDE_PT, -1)

    # Expert start marker
    ex_pt = (int(round(expert_start[0])), int(round(expert_start[1])))
    cv2.circle(canvas, ex_pt, 10, _COLOR_EXPERT_START, 2)
    cv2.putText(
        canvas, "EXPERT START",
        (ex_pt[0] + 13, ex_pt[1] + 4),
        cv2.FONT_HERSHEY_SIMPLEX, 0.55, _COLOR_EXPERT_START, 1,
    )

    # Learner start marker
    lx_pt = (int(round(learner_start[0])), int(round(learner_start[1])))
    cv2.circle(canvas, lx_pt, 10, _COLOR_LEARNER_START, 2)
    cv2.putText(
        canvas, "LEARNER START",
        (lx_pt[0] + 13, lx_pt[1] + 20),
        cv2.FONT_HERSHEY_SIMPLEX, 0.55, _COLOR_LEARNER_START, 1,
    )

    # Direction arrows (using first 10 points)
    if len(aligned_centerline) >= 10:
        a0 = (int(round(aligned_centerline[0][0])), int(round(aligned_centerline[0][1])))
        a9 = (int(round(aligned_centerline[9][0])), int(round(aligned_centerline[9][1])))
        cv2.arrowedLine(canvas, a0, a9, _COLOR_CENTERLINE, 2, tipLength=0.3)
    if len(learner_resampled) >= 10:
        l0 = (int(round(learner_resampled[0][0])), int(round(learner_resampled[0][1])))
        l9 = (int(round(learner_resampled[9][0])), int(round(learner_resampled[9][1])))
        cv2.arrowedLine(canvas, l0, l9, _COLOR_LEARNER, 2, tipLength=0.3)

    cv2.imwrite(str(output_path), canvas)


def _draw_frame0_alignment_overlay(
    frame: np.ndarray,
    aligned_centerline: list[tuple[float, float]],
    corridor_polygon: list[tuple[float, float]],
    left_edge: list[tuple[float, float]],
    right_edge: list[tuple[float, float]],
    learner_points: list[tuple[float, float]],
    expert_anchor_aligned: tuple[float, float],
    learner_anchor: tuple[float, float],
    annotation: str | None = None,
) -> None:
    poly_pts = _pts_i32(corridor_polygon)
    overlay = frame.copy()
    cv2.fillPoly(overlay, [poly_pts], _COLOR_CORRIDOR_FILL)
    cv2.addWeighted(overlay, _CORRIDOR_ALPHA, frame, 1 - _CORRIDOR_ALPHA, 0, dst=frame)

    if len(left_edge) >= 2:
        cv2.polylines(frame, [_pts_i32(left_edge)], False, _COLOR_EDGE, 1)
    if len(right_edge) >= 2:
        cv2.polylines(frame, [_pts_i32(right_edge)], False, _COLOR_EDGE, 1)
    if len(aligned_centerline) >= 2:
        cv2.polylines(frame, [_pts_i32(aligned_centerline)], False, _COLOR_CENTERLINE, 2)
    if len(learner_points) >= 2:
        cv2.polylines(frame, [_pts_i32(learner_points)], False, _COLOR_LEARNER, 2)

    expert_pt = (int(round(expert_anchor_aligned[0])), int(round(expert_anchor_aligned[1])))
    learner_pt = (int(round(learner_anchor[0])), int(round(learner_anchor[1])))
    cv2.circle(frame, expert_pt, 11, _COLOR_EXPERT_START, 2)
    cv2.circle(frame, learner_pt, 6, _COLOR_LEARNER_START, -1)
    cv2.putText(
        frame,
        "EXPERT/LEARNER ANCHOR",
        (expert_pt[0] + 12, max(24, expert_pt[1] - 10)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        _COLOR_CENTERLINE,
        2,
        cv2.LINE_AA,
    )
    if annotation:
        cv2.putText(
            frame,
            annotation,
            (20, 32),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            _COLOR_CENTERLINE,
            2,
            cv2.LINE_AA,
        )


def _render_frame0_preview(
    *,
    canvas_w: int,
    canvas_h: int,
    aligned_centerline: list[tuple[float, float]],
    left_edge: list[tuple[float, float]],
    right_edge: list[tuple[float, float]],
    corridor_polygon: list[tuple[float, float]],
    learner_points: list[tuple[float, float]],
    expert_anchor_aligned: tuple[float, float],
    learner_anchor: tuple[float, float],
    annotation: str,
    output_path: Path,
) -> None:
    canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
    _draw_frame0_alignment_overlay(
        canvas,
        aligned_centerline,
        corridor_polygon,
        left_edge,
        right_edge,
        learner_points,
        expert_anchor_aligned,
        learner_anchor,
        annotation,
    )
    cv2.imwrite(str(output_path), canvas)


def _render_frame0_overlay_video(
    *,
    learner_video_path: str | None,
    aligned_centerline: list[tuple[float, float]],
    corridor_polygon: list[tuple[float, float]],
    left_edge: list[tuple[float, float]],
    right_edge: list[tuple[float, float]],
    learner_points: list[tuple[float, float]],
    expert_anchor_aligned: tuple[float, float],
    learner_anchor: tuple[float, float],
    output_path: Path,
) -> None:
    cap: cv2.VideoCapture | None = None
    fps = 30.0
    w, h = 1920, 1080

    if learner_video_path:
        cap = cv2.VideoCapture(str(learner_video_path))
        if not cap.isOpened():
            cap = None

    if cap is not None:
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (w, h))

    try:
        if cap is not None:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                _draw_frame0_alignment_overlay(
                    frame,
                    aligned_centerline,
                    corridor_polygon,
                    left_edge,
                    right_edge,
                    learner_points,
                    expert_anchor_aligned,
                    learner_anchor,
                )
                writer.write(frame)
            cap.release()
        else:
            blank = np.zeros((h, w, 3), dtype=np.uint8)
            for _ in range(max(len(aligned_centerline), 30)):
                frame = blank.copy()
                _draw_frame0_alignment_overlay(
                    frame,
                    aligned_centerline,
                    corridor_polygon,
                    left_edge,
                    right_edge,
                    learner_points,
                    expert_anchor_aligned,
                    learner_anchor,
                )
                writer.write(frame)
    finally:
        writer.release()


def _render_overlay_video(
    *,
    learner_video_path: str | None,
    aligned_centerline: list[tuple[float, float]],
    corridor_polygon: list[tuple[float, float]],
    left_edge: list[tuple[float, float]],
    right_edge: list[tuple[float, float]],
    learner_full_points: list[tuple[float, float]],
    progress_checks: list[dict[str, Any]],
    output_path: Path,
) -> None:
    cap: cv2.VideoCapture | None = None
    fps = 30.0
    w, h = 1920, 1080

    if learner_video_path:
        cap = cv2.VideoCapture(str(learner_video_path))
        if not cap.isOpened():
            cap = None

    if cap is not None:
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (w, h))

    try:
        if cap is not None:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                _draw_aligned_corridor_overlay(
                    frame,
                    aligned_centerline,
                    corridor_polygon,
                    left_edge,
                    right_edge,
                    learner_full_points,
                    progress_checks,
                )
                writer.write(frame)
            cap.release()
        else:
            blank = np.zeros((h, w, 3), dtype=np.uint8)
            n_frames = max(len(aligned_centerline), 30)
            for _ in range(n_frames):
                frame = blank.copy()
                _draw_aligned_corridor_overlay(
                    frame,
                    aligned_centerline,
                    corridor_polygon,
                    left_edge,
                    right_edge,
                    learner_full_points,
                    progress_checks,
                )
                writer.write(frame)
    finally:
        writer.release()


# ──────────────────────────────────────────────────────────────────────────────
# Utility helpers
# ──────────────────────────────────────────────────────────────────────────────

def _video_dimensions(video_path: str | None) -> tuple[int, int]:
    if not video_path:
        return 1920, 1080
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return 1920, 1080
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    return (w, h) if w > 0 and h > 0 else (1920, 1080)


def _storage_url_for(path: Path) -> str | None:
    storage_root = BACKEND_ROOT / "storage"
    try:
        storage_key = path.resolve().relative_to(storage_root.resolve()).as_posix()
    except ValueError:
        return None
    return f"/storage/{quote(storage_key, safe='/')}"


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
