from __future__ import annotations

import bisect
import json
import math
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from app.services.angle.detector import extend_angle_line
from app.services.sam2_yolo.visualization import _convert_to_web_mp4

_BACKEND_ROOT = Path(__file__).resolve().parents[3]

_CORRIDOR_COLOR = (0, 0, 220)      # BGR red  (Component 1 — do not touch)

_FREEZE_TOTAL       = 45   # duplicate frames inserted at each trajectory peak
_FREEZE_ANIM        = 20   # frames over which trajectory annotation animates in
_ANGLE_FREEZE_TOTAL = 150  # duplicate frames inserted at each angle peak


# ── Fonts for annotation box ──────────────────────────────────────────────────

def _load_text_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = [
        r"C:\Windows\Fonts\arialbd.ttf",
        r"C:\Windows\Fonts\arial.ttf",
        r"C:\Windows\Fonts\seguisb.ttf",
        r"C:\Windows\Fonts\calibrib.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]
    for path in candidates:
        try:
            return ImageFont.truetype(path, size)
        except Exception:
            continue
    return ImageFont.load_default()


_FONT_ACCENT = _load_text_font(20)   # header line
_FONT_BODY   = _load_text_font(15)   # body text
_FONT_SMALL  = _load_text_font(12)   # smaller caption text


# ── Utilities ─────────────────────────────────────────────────────────────────

def _interp_to_all_frames(keyframes: dict[int, float], max_frame: int) -> list[float]:
    """Linearly interpolate a sparse {frame_index: value} dict to every frame 0..max_frame."""
    if not keyframes:
        return [0.0] * (max_frame + 1)
    sorted_keys = sorted(keyframes.keys())
    sorted_vals = [keyframes[k] for k in sorted_keys]
    result: list[float] = []
    for i in range(max_frame + 1):
        if i <= sorted_keys[0]:
            result.append(sorted_vals[0])
        elif i >= sorted_keys[-1]:
            result.append(sorted_vals[-1])
        else:
            pos = bisect.bisect_right(sorted_keys, i)
            lo_k, hi_k = sorted_keys[pos - 1], sorted_keys[pos]
            lo_v, hi_v = sorted_vals[pos - 1], sorted_vals[pos]
            t = (i - lo_k) / (hi_k - lo_k)
            result.append(lo_v * (1 - t) + hi_v * t)
    return result


def _extend_line_to_frame(
    p1: np.ndarray,
    p2: np.ndarray,
    width: int,
    height: int,
) -> tuple[tuple[int, int], tuple[int, int]]:
    """Return (top_pt, bottom_pt) where the line crosses y=0 and y=height."""
    x1, y1 = float(p1[0]), float(p1[1])
    x2, y2 = float(p2[0]), float(p2[1])
    line_dy = y2 - y1
    if abs(line_dy) < 1e-6:
        y = int(round(y1))
        return (0, y), (width, y)
    slope = (x2 - x1) / line_dy
    xt = x1 + slope * (0.0 - y1)
    xb = x1 + slope * (float(height) - y1)
    return (int(round(xt)), 0), (int(round(xb)), height)


def _draw_dashed_segment(
    img: np.ndarray,
    p1: tuple[float, float],
    p2: tuple[float, float],
    color: tuple[int, int, int],
    thickness: int,
    dash: int = 12,
    gap: int = 8,
) -> None:
    x1, y1 = float(p1[0]), float(p1[1])
    x2, y2 = float(p2[0]), float(p2[1])
    length = math.hypot(x2 - x1, y2 - y1)
    if length < 1:
        return
    ux, uy = (x2 - x1) / length, (y2 - y1) / length
    pos, drawing = 0.0, True
    while pos < length:
        seg = dash if drawing else gap
        end = min(pos + seg, length)
        if drawing:
            cv2.line(
                img,
                (int(round(x1 + ux * pos)),  int(round(y1 + uy * pos))),
                (int(round(x1 + ux * end)),  int(round(y1 + uy * end))),
                color, thickness, cv2.LINE_AA,
            )
        pos = end
        drawing = not drawing


def _draw_dashed_polyline(
    img: np.ndarray,
    points: list[tuple[float, float]],
    color: tuple[int, int, int],
    thickness: int,
    dash: int = 12,
    gap: int = 8,
) -> None:
    for i in range(len(points) - 1):
        _draw_dashed_segment(img, points[i], points[i + 1], color, thickness, dash, gap)


def _total_turning_angle_deg(pts: list[tuple[float, float]]) -> float:
    """Sum of turning angles in degrees at each interior vertex of a polyline."""
    total = 0.0
    for i in range(1, len(pts) - 1):
        dx1, dy1 = pts[i][0] - pts[i - 1][0], pts[i][1] - pts[i - 1][1]
        dx2, dy2 = pts[i + 1][0] - pts[i][0],  pts[i + 1][1] - pts[i][1]
        len1 = math.hypot(dx1, dy1)
        len2 = math.hypot(dx2, dy2)
        if len1 < 1e-9 or len2 < 1e-9:
            continue
        cos_a = max(-1.0, min(1.0, (dx1 * dx2 + dy1 * dy2) / (len1 * len2)))
        total += math.degrees(math.acos(cos_a))
    return total


def _project_onto_polyline(
    px: float,
    py: float,
    poly: list[tuple[float, float]],
    cum_len: list[float],
) -> float:
    """Return arc-length of the orthogonal projection of (px, py) onto poly."""
    best_arc = 0.0
    best_dist = float("inf")
    for i in range(len(poly) - 1):
        ax, ay = poly[i]
        bx, by = poly[i + 1]
        abx, aby = bx - ax, by - ay
        ab_len_sq = abx * abx + aby * aby
        t = ((px - ax) * abx + (py - ay) * aby) / ab_len_sq if ab_len_sq > 1e-12 else 0.0
        t = max(0.0, min(1.0, t))
        cx, cy = ax + t * abx, ay + t * aby
        dist = math.hypot(px - cx, py - cy)
        if dist < best_dist:
            best_dist = dist
            best_arc = cum_len[i] + t * math.hypot(abx, aby)
    return best_arc


def _poly_pts_up_to_arc(
    poly: list[tuple[float, float]],
    cum_len: list[float],
    target_arc: float,
) -> list[tuple[float, float]]:
    """Return the sub-polyline of poly covering arc-length 0..target_arc."""
    pts: list[tuple[float, float]] = []
    for i, pt in enumerate(poly):
        if cum_len[i] <= target_arc:
            pts.append(pt)
        else:
            if i > 0:
                seg_arc = cum_len[i] - cum_len[i - 1]
                if seg_arc > 0:
                    t = (target_arc - cum_len[i - 1]) / seg_arc
                    px0, py0 = poly[i - 1]
                    pts.append((px0 + t * (pt[0] - px0), py0 + t * (pt[1] - py0)))
            break
    return pts


def _expert_x_at_y(
    pts: list[tuple[float, float]],
    target_y: float,
) -> float | None:
    """Return the X coordinate of a polyline at target_y, or None if out of range."""
    for i in range(len(pts) - 1):
        x0, y0 = pts[i]
        x1, y1 = pts[i + 1]
        if (y0 <= target_y <= y1) or (y1 <= target_y <= y0):
            dy = y1 - y0
            t = (target_y - y0) / dy if abs(dy) > 1e-9 else 0.5
            return x0 + t * (x1 - x0)
    return None


def _draw_arrow_alpha(
    frame: np.ndarray,
    pt1: tuple[int, int],
    pt2: tuple[int, int],
    color: tuple[int, int, int],
    alpha: float,
    thickness: int = 2,
    tip_length: float = 0.25,
) -> None:
    overlay = frame.copy()
    cv2.arrowedLine(overlay, pt1, pt2, color, thickness, cv2.LINE_AA, tipLength=tip_length)
    cv2.addWeighted(overlay, alpha, frame, 1.0 - alpha, 0, frame)


def _render_freeze_annotation(
    frame: np.ndarray,
    lx: float,
    ly: float,
    ex: float,
    ey: float,
    t: float,       # animation progress 0..1
    width: int,
    height: int,
) -> None:
    """Draw peak-deviation freeze annotation onto frame in-place at progress t."""
    ilx, ily = int(round(lx)), int(round(ly))

    # 1. Growing dashed white line from learner tip toward expert position
    tip_x = int(round(lx + (ex - lx) * t))
    tip_y = int(round(ly + (ey - ly) * t))
    if (ilx, ily) != (tip_x, tip_y):
        _draw_dashed_segment(frame, (ilx, ily), (tip_x, tip_y), (255, 255, 255), 2, 10, 6)

    # 2. Pulsing red dot at learner blade tip
    pulse = 0.5 + 0.5 * math.sin(t * math.pi * 4)   # two pulses over animation
    dot_r = int(round(8 + 5 * pulse))
    dot_ov = frame.copy()
    cv2.circle(dot_ov, (ilx, ily), dot_r, (0, 0, 255), -1, cv2.LINE_AA)
    cv2.addWeighted(dot_ov, 0.88, frame, 0.12, 0, frame)

    # 3. Annotation box sliding in from right with ease-out
    ease = 1.0 - (1.0 - t) ** 2
    box_w, box_h = 310, 78
    margin = 20
    target_x = width - box_w - margin
    box_x = int(round(width + (target_x - width) * ease))
    box_y = max(margin, min(height - box_h - margin, ily - box_h // 2))

    # Dark semi-transparent background panel
    panel_ov = frame.copy()
    cv2.rectangle(panel_ov, (box_x, box_y), (box_x + box_w, box_y + box_h), (20, 20, 20), -1)
    cv2.addWeighted(panel_ov, 0.82, frame, 0.18, 0, frame)
    # Amber accent border
    cv2.rectangle(frame, (box_x, box_y), (box_x + box_w, box_y + box_h), (0, 165, 255), 1)

    # Text via PIL  (PIL uses RGB; frame is BGR)
    pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil)
    tx = box_x + 14
    draw.text((tx, box_y + 10), "PEAK DEVIATION",           font=_FONT_ACCENT, fill=(255, 180, 0))
    draw.text((tx, box_y + 38), "Scissors strayed from the", font=_FONT_BODY,   fill=(255, 255, 255))
    draw.text((tx, box_y + 56), "expert cutting path",       font=_FONT_BODY,   fill=(255, 255, 255))
    frame[:] = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)

    # 4. Thin red connecting line from dot to left edge of annotation box
    if ease > 0.02:
        cv2.line(frame, (ilx, ily), (box_x, box_y + box_h // 2), (0, 0, 200), 1, cv2.LINE_AA)


def _line_intersection(
    ox1: float, oy1: float, dx1: float, dy1: float,
    ox2: float, oy2: float, dx2: float, dy2: float,
) -> tuple[float, float]:
    # solve: (ox1 + t*dx1, oy1 + t*dy1) = (ox2 + s*dx2, oy2 + s*dy2)
    denom = dx1 * dy2 - dy1 * dx2
    if abs(denom) < 1e-6:  # parallel
        return (ox1 + ox2) / 2, (oy1 + oy2) / 2
    t = ((ox2 - ox1) * dy2 - (oy2 - oy1) * dx2) / denom
    ix = ox1 + t * dx1
    iy = oy1 + t * dy1
    return ix, iy


def _dir_to_cv2_ellipse_angle(vx: float, vy: float) -> float:
    """Convert line direction (vx, vy) to a cv2.ellipse angle (CW from +x axis, degrees)."""
    if vy > 0:
        vx, vy = -vx, -vy
    return math.degrees(math.atan2(vx, -vy)) - 90.0


def _render_angle_freeze(
    frame: np.ndarray,
    cx: float,
    cy: float,
    ex: float,
    ey: float,
    learner_angle_deg: float,
    expert_angle_deg: float,
    angle_diff_deg: float,
    anim_i: int,                         # 0..89
    width: int,
    height: int,
) -> None:
    """Draw angle-error freeze annotation using direct line drawing.

    Phase 1 (0-19):  learner angle line fades in (white)   — origin: learner peak location
    Phase 2 (20-39): expert angle line fades in (cyan)     — origin: expert trajectory tip
    Phase 3 (40-59): arc + annotation box fade in
    Hold   (60-89):  everything fully drawn
    """
    ox, oy = int(round(cx)), int(round(cy))

    p_l = min(anim_i / 19.0, 1.0)
    p_e = 0.0 if anim_i < 20 else min((anim_i - 20) / 19.0, 1.0)
    t3  = 0.0 if anim_i < 40 else (1.0 if anim_i >= 60 else (anim_i - 40) / 19.0)

    # ── Learner angle line (white) ────────────────────────────────────────────
    if p_l > 0:
        l_start, l_end = extend_angle_line(learner_angle_deg, cx, cy, width, height)
        overlay = frame.copy()
        cv2.line(overlay, l_start, l_end, (255, 255, 255), 3, cv2.LINE_AA)
        cv2.addWeighted(overlay, p_l * 0.85, frame, 1.0 - p_l * 0.85, 0, frame)

    # ── Expert angle line (cyan, BGR = 255, 255, 0) ───────────────────────────
    if p_e > 0:
        e_start, e_end = extend_angle_line(expert_angle_deg, ex, ey, width, height)
        overlay = frame.copy()
        cv2.line(overlay, e_start, e_end, (255, 255, 0), 3, cv2.LINE_AA)
        cv2.addWeighted(overlay, p_e * 0.85, frame, 1.0 - p_e * 0.85, 0, frame)

    # ── Phase 3: arc + annotation ─────────────────────────────────────────────
    if t3 > 0:
        l_rad = math.radians(learner_angle_deg)
        e_rad = math.radians(expert_angle_deg)

        # Direction from clipped endpoints — same geometry used to draw the lines on screen
        l_p1, l_p2 = extend_angle_line(learner_angle_deg, cx, cy, width, height)
        e_p1, e_p2 = extend_angle_line(expert_angle_deg, ex, ey, width, height)
        ldx = l_p2[0] - l_p1[0]
        ldy = l_p2[1] - l_p1[1]
        edx = e_p2[0] - e_p1[0]
        edy = e_p2[1] - e_p1[1]
        ix, iy = _line_intersection(l_p1[0], l_p1[1], ldx, ldy, e_p1[0], e_p1[1], edx, edy)
        if not (0 <= ix <= width and 0 <= iy <= height):
            ix, iy = (cx + ex) / 2.0, (cy + ey) / 2.0
        print(f"[ANGLE ARC] learner_origin=({cx:.0f},{cy:.0f}) expert_origin=({ex:.0f},{ey:.0f}) intersection=({ix:.0f},{iy:.0f})", flush=True)
        i_pt = (int(round(ix)), int(round(iy)))

        learner_dir = (math.cos(l_rad), math.sin(l_rad))
        expert_dir  = (math.cos(e_rad), math.sin(e_rad))
        e_cv2 = _dir_to_cv2_ellipse_angle(*expert_dir)
        l_cv2 = _dir_to_cv2_ellipse_angle(*learner_dir)
        diff = l_cv2 - e_cv2
        if diff > 180:
            l_cv2 -= 360.0
        elif diff < -180:
            l_cv2 += 360.0
        cv2.ellipse(frame, i_pt, (60, 60), 0.0,
                    e_cv2, l_cv2, (0, 165, 255), 2, cv2.LINE_AA)
        mid_cv2   = (e_cv2 + l_cv2) / 2.0
        arc_mid_x = int(round(ix + 60.0 * math.cos(math.radians(mid_cv2))))
        arc_mid_y = int(round(iy + 60.0 * math.sin(math.radians(mid_cv2))))
        lbl_x     = int(round(ix + 78.0 * math.cos(math.radians(mid_cv2))))
        lbl_y     = int(round(iy + 78.0 * math.sin(math.radians(mid_cv2))))

        ease     = 1.0 - (1.0 - t3) ** 2
        box_w, box_h = 370, 96
        margin   = 20
        target_x = width - box_w - margin
        box_x    = int(round(width + (target_x - width) * ease))
        box_y    = max(margin, min(height - box_h - margin, oy - box_h // 2))

        panel_ov = frame.copy()
        cv2.rectangle(panel_ov, (box_x, box_y), (box_x + box_w, box_y + box_h), (20, 20, 20), -1)
        cv2.addWeighted(panel_ov, 0.82, frame, 0.18, 0, frame)
        cv2.rectangle(frame, (box_x, box_y), (box_x + box_w, box_y + box_h), (0, 165, 255), 1)

        pil  = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pil)
        tx   = box_x + 14
        draw.text((tx, box_y + 8),
                  "PEAK ANGLE DEVIATION",
                  font=_FONT_ACCENT, fill=(255, 180, 0))
        draw.text((tx, box_y + 36),
                  f"Scissors tilted {angle_diff_deg:.0f}° off the correct cutting angle",
                  font=_FONT_BODY, fill=(255, 255, 255))
        draw.text((tx, box_y + 62),
                  "This will push the cut off the expert path",
                  font=_FONT_SMALL, fill=(255, 255, 255))
        draw.text((lbl_x, lbl_y), f"{angle_diff_deg:.0f}°", font=_FONT_SMALL, fill=(255, 180, 0))
        frame[:] = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)

        if ease > 0.02:
            cv2.line(frame, (arc_mid_x, arc_mid_y), (box_x, box_y + box_h // 2),
                     (0, 165, 255), 1, cv2.LINE_AA)


# ── Main renderer ─────────────────────────────────────────────────────────────

def render_visualization(
    learner_video_path: str,
    expert_id: str,
    run_id: str,
    output_dir: str,
    fps: int = 30,
) -> str:
    """Render learner video with corridor lines + trajectory error overlay.

    Returns the absolute path to the saved visualization.mp4.
    """
    metrics_path = (
        _BACKEND_ROOT / "storage" / "outputs" / "sam2_yolo" / "experts" / expert_id / "metrics.json"
    )
    corridor_path = (
        _BACKEND_ROOT / "storage" / "evaluation" / run_id / "trajectory" / "aligned_corridor.json"
    )
    traj_path = (
        _BACKEND_ROOT / "storage" / "outputs" / "sam2_yolo" / "experts" / expert_id / "trajectory_smoothed.json"
    )
    dtw_path = (
        _BACKEND_ROOT / "storage" / "evaluation" / run_id / "angle" / "dtw_alignment.json"
    )
    errors_path = (
        _BACKEND_ROOT / "storage" / "evaluation" / run_id / "score" / "unified_errors.json"
    )

    with open(metrics_path) as f:
        metrics = json.load(f)
    with open(corridor_path) as f:
        corridor = json.load(f)
    with open(traj_path) as f:
        traj_data = json.load(f)

    # ── Component 1: Corridor line setup (DO NOT TOUCH) ───────────────────────
    fitted    = metrics["trajectory_metrics"]["fitted_line"]
    direction = np.array(fitted["direction"],    dtype=np.float64)
    start_pt  = np.array(fitted["start_point"], dtype=np.float64)
    end_pt    = np.array(fitted["end_point"],   dtype=np.float64)
    normal    = np.array([-direction[1], direction[0]], dtype=np.float64)

    dx = float(corridor["translation"]["dx"])
    dy = float(corridor["translation"]["dy"])
    translation = np.array([dx, dy], dtype=np.float64)

    left_start  = start_pt + 80.0 * normal + translation
    left_end    = end_pt   + 80.0 * normal + translation
    right_start = start_pt - 80.0 * normal + translation
    right_end   = end_pt   - 80.0 * normal + translation

    # ── Expert smoothed trajectory ────────────────────────────────────────────
    traj_x_kf: dict[int, float] = {}
    traj_y_kf: dict[int, float] = {}
    for pt in traj_data.get("points", []):
        fi = int(pt["frame_index"])
        traj_x_kf[fi] = float(pt["smoothed_x"])
        traj_y_kf[fi] = float(pt["smoothed_y"])

    expert_max_frame = max(traj_x_kf.keys()) if traj_x_kf else 0
    traj_x_all = _interp_to_all_frames(traj_x_kf, expert_max_frame)
    traj_y_all = _interp_to_all_frames(traj_y_kf, expert_max_frame)

    # Sparse keyframe points pre-translated into learner pixel space
    expert_kf_sorted = sorted(traj_x_kf.keys())
    expert_kf_pts: dict[int, tuple[float, float]] = {
        fi: (traj_x_kf[fi] + dx, traj_y_kf[fi] + dy) for fi in expert_kf_sorted
    }

    # ── DTW alignment: learner_frame → expert_frame + angles ─────────────────
    learner_to_expert_kf: dict[int, float] = {}
    dtw_angles_by_learner: dict[int, tuple[float, float]] = {}   # lf → (learner_angle, expert_angle)
    if dtw_path.is_file():
        with open(dtw_path) as f:
            dtw_data = json.load(f)
        for m in dtw_data.get("matches", []):
            lf = int(m["learner_frame_index"])
            learner_to_expert_kf[lf] = float(m["expert_frame_index"])
            dtw_angles_by_learner[lf] = (float(m["learner_angle"]), float(m["expert_angle"]))

    # ── Frame checks: learner blade tip, outside flag, deviation ──────────────
    frame_checks = corridor.get("frame_checks", [])
    fc_by_frame: dict[int, dict] = {int(fc["frame_index"]): fc for fc in frame_checks}
    fc_sorted = sorted(fc_by_frame.keys())

    def _nearest_fc(fidx: int) -> dict | None:
        if not fc_sorted:
            return None
        pos = bisect.bisect_left(fc_sorted, fidx)
        candidates: list[int] = []
        if pos < len(fc_sorted):
            candidates.append(fc_sorted[pos])
        if pos > 0:
            candidates.append(fc_sorted[pos - 1])
        return fc_by_frame[min(candidates, key=lambda f: abs(f - fidx))]

    # ── Spatial rescaling: stretch expert path to match learner path length ─────
    def _poly_length(pts: list[tuple[float, float]]) -> float:
        return sum(
            math.hypot(pts[i + 1][0] - pts[i][0], pts[i + 1][1] - pts[i][1])
            for i in range(len(pts) - 1)
        )

    expert_pts_list = [expert_kf_pts[fi] for fi in expert_kf_sorted]
    learner_pts_list = [
        (float(fc_by_frame[f]["learner_x"]), float(fc_by_frame[f]["learner_y"]))
        for f in fc_sorted
    ]
    expert_length = _poly_length(expert_pts_list)
    learner_length = _poly_length(learner_pts_list)
    path_scale = (learner_length / expert_length) if expert_length > 1e-6 else 1.0

    if path_scale != 1.0 and expert_kf_sorted:
        ox = traj_x_kf[expert_kf_sorted[0]]   # raw-space origin (first keyframe)
        oy = traj_y_kf[expert_kf_sorted[0]]
        traj_x_all = [ox + path_scale * (x - ox) for x in traj_x_all]
        traj_y_all = [oy + path_scale * (y - oy) for y in traj_y_all]
        expert_kf_pts = {fi: (traj_x_all[fi] + dx, traj_y_all[fi] + dy) for fi in expert_kf_sorted}

    # ── Reference polyline for projection-based line growth ───────────────────
    ref_poly: list[tuple[float, float]] = [expert_kf_pts[fi] for fi in expert_kf_sorted]
    # Collapse to a straight line when the path barely curves (< 30° total turn)
    if len(ref_poly) >= 3 and _total_turning_angle_deg(ref_poly) < 30.0:
        ref_poly = [ref_poly[0], ref_poly[-1]]
    ref_cum_len: list[float] = [0.0]
    for i in range(len(ref_poly) - 1):
        ref_cum_len.append(
            ref_cum_len[-1]
            + math.hypot(ref_poly[i + 1][0] - ref_poly[i][0], ref_poly[i + 1][1] - ref_poly[i][1])
        )
    ref_total_len = ref_cum_len[-1]

    # ── Trajectory errors → freeze events ─────────────────────────────────────
    traj_errors: list[dict] = []
    angle_errors_raw: list[dict] = []
    if errors_path.is_file():
        with open(errors_path) as f:
            _errors_data = json.load(f)
        traj_errors       = _errors_data.get("trajectory_errors", [])
        angle_errors_raw  = _errors_data.get("angle_errors", [])

    lf_dtw_sorted = sorted(learner_to_expert_kf.keys())

    def _nearest_expert_fi(learner_fidx: int) -> int:
        if not lf_dtw_sorted:
            return 0
        pos = bisect.bisect_left(lf_dtw_sorted, learner_fidx)
        candidates: list[int] = []
        if pos < len(lf_dtw_sorted):
            candidates.append(lf_dtw_sorted[pos])
        if pos > 0:
            candidates.append(lf_dtw_sorted[pos - 1])
        nearest_lf = min(candidates, key=lambda f: abs(f - learner_fidx))
        return min(int(round(learner_to_expert_kf[nearest_lf])), expert_max_frame)

    # For each error window: find the peak-deviation frame and store freeze data.
    # freeze_events maps original_frame_index → (lx, ly, expert_x, expert_y)
    freeze_events: dict[int, tuple[float, float, float, float]] = {}
    for err in traj_errors:
        f_start, f_end = int(err["frame_start"]), int(err["frame_end"])
        window_fcs = [fc_by_frame[f] for f in fc_sorted if f_start <= f <= f_end]
        if not window_fcs:
            fc = _nearest_fc((f_start + f_end) // 2)
            if fc:
                window_fcs = [fc]
        if not window_fcs:
            continue
        peak_fc = max(window_fcs, key=lambda fc: float(fc.get("deviation_px", 0)))
        peak_frame = int(peak_fc["frame_index"])
        lx_p = float(peak_fc["learner_x"])
        ly_p = float(peak_fc["learner_y"])
        efi   = _nearest_expert_fi(peak_frame)
        ex_p  = traj_x_all[efi] + dx
        ey_p  = traj_y_all[efi] + dy
        freeze_events[peak_frame] = (lx_p, ly_p, ex_p, ey_p)

    # ── Angle errors → angle freeze events ───────────────────────────────────
    _dtw_lf_sorted = sorted(dtw_angles_by_learner.keys())

    def _nearest_dtw_angles(learner_fidx: int) -> tuple[float, float]:
        """Return (learner_angle, expert_angle) for the DTW match nearest to learner_fidx."""
        if not _dtw_lf_sorted:
            return 0.0, 0.0
        pos = bisect.bisect_left(_dtw_lf_sorted, learner_fidx)
        cands: list[int] = []
        if pos < len(_dtw_lf_sorted):
            cands.append(_dtw_lf_sorted[pos])
        if pos > 0:
            cands.append(_dtw_lf_sorted[pos - 1])
        nearest = min(cands, key=lambda f: abs(f - learner_fidx))
        return dtw_angles_by_learner[nearest]

    angle_freeze_events: dict[int, dict] = {}
    for err in angle_errors_raw:
        peak_frame  = int(err["peak_frame"])
        lx_a        = float(err["peak_location"]["x"])
        ly_a        = float(err["peak_location"]["y"])
        angle_diff  = float(err["peak_angle_diff_deg"])
        learner_ang, expert_ang = _nearest_dtw_angles(peak_frame)
        angle_freeze_events[peak_frame] = {
            "lx": lx_a, "ly": ly_a,
            "learner_angle_deg": learner_ang,
            "expert_angle_deg":  expert_ang,
            "angle_diff_deg":    angle_diff,
        }

    # ── Open learner video ────────────────────────────────────────────────────
    cap = cv2.VideoCapture(learner_video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {learner_video_path}")

    video_fps          = cap.get(cv2.CAP_PROP_FPS) or float(fps)
    width              = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height             = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    learner_total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
    learner_max_frame  = max(learner_total_frames - 1, 0)

    # ── Angle JSON sources ────────────────────────────────────────────────────
    _learner_angles_json = (
        _BACKEND_ROOT / "storage" / "evaluation" / run_id / "angle" / "angles.json"
    )
    _expert_angles_json = (
        _BACKEND_ROOT / "storage" / "outputs" / "angles" / "expert" / expert_id / "angles.json"
    )

    learner_angle_by_frame: dict[int, dict] = {}
    expert_angle_by_frame: dict[int, dict] = {}

    if _learner_angles_json.is_file():
        with open(_learner_angles_json) as _f:
            for _fr in json.load(_f).get("frames", []):
                if _fr.get("valid_line") and _fr.get("line_angle") is not None:
                    learner_angle_by_frame[int(_fr["frame_index"])] = _fr
    else:
        print(f"[angle freeze] WARNING: {_learner_angles_json} not found", flush=True)

    if _expert_angles_json.is_file():
        with open(_expert_angles_json) as _f:
            for _fr in json.load(_f).get("frames", []):
                if _fr.get("valid_line") and _fr.get("line_angle") is not None:
                    expert_angle_by_frame[int(_fr["frame_index"])] = _fr
    else:
        print(f"[angle freeze] WARNING: {_expert_angles_json} not found", flush=True)

    angle_freeze_enabled = bool(angle_freeze_events)

    # Dense interpolation of DTW mapping for smooth expert-frame lookup
    learner_to_expert_all = _interp_to_all_frames(learner_to_expert_kf, learner_max_frame)

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tmp_path     = out_dir / "visualization.tmp.mp4"
    web_tmp_path = out_dir / "visualization.web.tmp.mp4"
    out_path     = out_dir / "visualization.mp4"

    writer = cv2.VideoWriter(
        str(tmp_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        max(video_fps, 1.0),
        (width, height),
    )
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"Could not create output video at {tmp_path}")

    lp1, lp2 = _extend_line_to_frame(left_start,  left_end,  width, height)
    rp1, rp2 = _extend_line_to_frame(right_start, right_end, width, height)

    draw_progress = 0.0       # monotonically increasing expert-line draw progress [0..1]
    last_arrow_frame = -25    # frame index of the most recently added arrow
    accumulated_arrows: list[tuple[tuple[int, int], tuple[int, int]]] = []
    frame_idx = 0
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # ── Component 1: Red corridor lines (unchanged logic) ──────────
            overlay = frame.copy()
            cv2.line(overlay, lp1, lp2, _CORRIDOR_COLOR, 3, cv2.LINE_AA)
            cv2.line(overlay, rp1, rp2, _CORRIDOR_COLOR, 3, cv2.LINE_AA)
            frame = cv2.addWeighted(overlay, 0.8, frame, 0.2, 0)

            # ── Component 2a: Expert trajectory (dashed yellow, growing) ───
            # Project learner blade tip onto the reference polyline to get
            # noise-free forward progress (zigzag oscillations are ignored).
            fc_cur = _nearest_fc(frame_idx)
            if fc_cur is not None and ref_total_len > 1e-6:
                proj_arc = _project_onto_polyline(
                    float(fc_cur["learner_x"]), float(fc_cur["learner_y"]),
                    ref_poly, ref_cum_len,
                )
                raw_progress = proj_arc / ref_total_len
            else:
                raw_progress = draw_progress
            # Fix 3: clamp — never go backwards
            draw_progress = min(max(draw_progress, raw_progress), 1.0)

            pts = _poly_pts_up_to_arc(ref_poly, ref_cum_len, draw_progress * ref_total_len)
            if len(pts) >= 2:
                line_ov = frame.copy()
                _draw_dashed_polyline(line_ov, pts, (0, 255, 255), 2)   # yellow
                cv2.addWeighted(line_ov, 0.9, frame, 0.1, 0, frame)

            # ── Component 2b: Correction arrows (horizontal, outside frames only) ─
            if frame_idx % 7 == 0:
                fc = _nearest_fc(frame_idx)
                if (fc and fc.get("outside", False)
                        and len(accumulated_arrows) < 12
                        and frame_idx - last_arrow_frame >= 60):
                    ilx = int(round(float(fc["learner_x"])))
                    ily = int(round(float(fc["learner_y"])))
                    # Find where the expert drawn line is at this Y — horizontal arrow
                    ex_at_y = _expert_x_at_y(pts, float(fc["learner_y"]))
                    if ex_at_y is not None:
                        iex = int(round(ex_at_y))
                        # Both endpoints share ily → perfectly horizontal arrow
                        accumulated_arrows.append(((ilx, ily), (iex, ily)))
                        last_arrow_frame = frame_idx

            # Redraw all accumulated arrows so they persist for the rest of the video
            for arr_pt1, arr_pt2 in accumulated_arrows:
                _draw_arrow_alpha(frame, arr_pt1, arr_pt2, (210, 60, 10), 0.65)

            writer.write(frame)

            # ── Peak deviation freeze ──────────────────────────────────────
            if frame_idx in freeze_events:
                lx_f, ly_f, ex_f, ey_f = freeze_events[frame_idx]
                # Capture the actual tip of the expert yellow line as drawn on this frame
                expert_tip_x, expert_tip_y = pts[-1] if pts else (ex_f, ey_f)
                # Always place an arrow at the peak frame regardless of spacing
                ex_at_peak_y = _expert_x_at_y(pts, ly_f)
                if ex_at_peak_y is not None:
                    accumulated_arrows.append(
                        ((int(round(lx_f)), int(round(ly_f))), (int(round(ex_at_peak_y)), int(round(ly_f))))
                    )
                for anim_i in range(_FREEZE_TOTAL):
                    frozen = frame.copy()
                    t = min(anim_i, _FREEZE_ANIM - 1) / (_FREEZE_ANIM - 1)
                    _render_freeze_annotation(frozen, lx_f, ly_f, expert_tip_x, expert_tip_y, t, width, height)
                    writer.write(frozen)

            # ── Component 3: Angle error freeze ───────────────────────────
            if frame_idx in angle_freeze_events and angle_freeze_enabled:
                ev = angle_freeze_events[frame_idx]
                lx_a, ly_a = ev["lx"], ev["ly"]

                # Use line_center from learner angles JSON for precise draw origin
                l_fr = learner_angle_by_frame.get(frame_idx)
                if l_fr is not None and isinstance(l_fr.get("line_center"), dict):
                    lc = l_fr["line_center"]
                    cx_a = float(lc.get("x", lx_a))
                    cy_a = float(lc.get("y", ly_a))
                else:
                    cx_a, cy_a = lx_a, ly_a

                etx_a, ety_a = pts[-1] if pts else (cx_a, cy_a)

                for anim_i in range(_ANGLE_FREEZE_TOTAL):
                    frozen = frame.copy()
                    _render_angle_freeze(
                        frozen, cx_a, cy_a, etx_a, ety_a,
                        ev["learner_angle_deg"], ev["expert_angle_deg"],
                        ev["angle_diff_deg"],
                        anim_i, width, height,
                    )
                    writer.write(frozen)

            frame_idx += 1
    finally:
        cap.release()
        writer.release()

    # Re-encode to H.264 for browser playback
    if _convert_to_web_mp4(tmp_path, web_tmp_path):
        tmp_path.unlink(missing_ok=True)
        web_tmp_path.replace(out_path)
    else:
        tmp_path.replace(out_path)

    return str(out_path)
