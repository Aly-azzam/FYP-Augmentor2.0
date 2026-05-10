from __future__ import annotations

import bisect
import json
from pathlib import Path

import cv2
import numpy as np

from app.services.sam2_yolo.visualization import _convert_to_web_mp4

_BACKEND_ROOT = Path(__file__).resolve().parents[3]

_CORRIDOR_COLOR = (0, 0, 220)    # BGR = red


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


def render_visualization(
    learner_video_path: str,
    expert_id: str,
    run_id: str,
    output_dir: str,
    fps: int = 30,
) -> str:
    """Render learner video with corridor lines + YOLO bbox crop ghost overlay.

    Returns the absolute path to the saved visualization.mp4.
    """
    metrics_path = (
        _BACKEND_ROOT / "storage" / "outputs" / "sam2_yolo" / "experts" / expert_id / "metrics.json"
    )
    corridor_path = (
        _BACKEND_ROOT / "storage" / "evaluation" / run_id / "trajectory" / "aligned_corridor.json"
    )
    raw_path = (
        _BACKEND_ROOT / "storage" / "outputs" / "sam2_yolo" / "experts" / expert_id / "raw.json"
    )
    metadata_path = (
        _BACKEND_ROOT / "storage" / "outputs" / "sam2_yolo" / "experts" / expert_id / "metadata.json"
    )

    with open(metrics_path) as f:
        metrics = json.load(f)
    with open(corridor_path) as f:
        corridor = json.load(f)
    with open(raw_path) as f:
        raw = json.load(f)
    with open(metadata_path) as f:
        metadata = json.load(f)

    # ── Corridor line setup ────────────────────────────────────────────────────
    fitted = metrics["trajectory_metrics"]["fitted_line"]
    direction = np.array(fitted["direction"], dtype=np.float64)
    start_pt = np.array(fitted["start_point"], dtype=np.float64)
    end_pt = np.array(fitted["end_point"], dtype=np.float64)

    normal = np.array([-direction[1], direction[0]], dtype=np.float64)

    dx = float(corridor["translation"]["dx"])
    dy = float(corridor["translation"]["dy"])
    translation = np.array([dx, dy], dtype=np.float64)

    left_start = start_pt + 80.0 * normal + translation
    left_end = end_pt + 80.0 * normal + translation
    right_start = start_pt - 80.0 * normal + translation
    right_end = end_pt - 80.0 * normal + translation

    # ── Build interpolated bbox per expert frame ───────────────────────────────
    bbox_x1_kf: dict[int, float] = {}
    bbox_y1_kf: dict[int, float] = {}
    bbox_x2_kf: dict[int, float] = {}
    bbox_y2_kf: dict[int, float] = {}

    for fr in raw.get("frames", []):
        bbox = fr.get("mask_bbox")
        if bbox is None:
            continue
        fi = int(fr["frame_index"])
        bbox_x1_kf[fi] = float(bbox[0])
        bbox_y1_kf[fi] = float(bbox[1])
        bbox_x2_kf[fi] = float(bbox[2])
        bbox_y2_kf[fi] = float(bbox[3])

    expert_max_frame = max(bbox_x1_kf.keys()) if bbox_x1_kf else 0
    expert_total_frames = expert_max_frame + 1

    bbox_x1_all = _interp_to_all_frames(bbox_x1_kf, expert_max_frame)
    bbox_y1_all = _interp_to_all_frames(bbox_y1_kf, expert_max_frame)
    bbox_x2_all = _interp_to_all_frames(bbox_x2_kf, expert_max_frame)
    bbox_y2_all = _interp_to_all_frames(bbox_y2_kf, expert_max_frame)

    # ── Open expert video ─────────────────────────────────────────────────────
    expert_video_path = metadata["source_video_path"]
    expert_cap = cv2.VideoCapture(str(expert_video_path))
    if not expert_cap.isOpened():
        raise RuntimeError(f"Could not open expert video: {expert_video_path}")

    # ── Open learner video ────────────────────────────────────────────────────
    cap = cv2.VideoCapture(learner_video_path)
    if not cap.isOpened():
        expert_cap.release()
        raise RuntimeError(f"Could not open learner video: {learner_video_path}")

    video_fps = cap.get(cv2.CAP_PROP_FPS) or float(fps)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    learner_total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tmp_path = out_dir / "visualization.tmp.mp4"
    web_tmp_path = out_dir / "visualization.web.tmp.mp4"
    out_path = out_dir / "visualization.mp4"

    writer = cv2.VideoWriter(
        str(tmp_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        max(video_fps, 1.0),
        (width, height),
    )
    if not writer.isOpened():
        cap.release()
        expert_cap.release()
        raise RuntimeError(f"Could not create output video at {tmp_path}")

    lp1, lp2 = _extend_line_to_frame(left_start, left_end, width, height)
    rp1, rp2 = _extend_line_to_frame(right_start, right_end, width, height)

    # Sequential expert cap state — never seek, only advance forward
    expert_pos = 0          # frame index of the last frame read from expert_cap
    expert_frame = None     # last successfully read expert frame

    # Prime the first expert frame
    ret_e, expert_frame = expert_cap.read()
    if not ret_e:
        expert_frame = None

    frame_idx = 0
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Overlay 1: corridor lines (alpha 0.8)
            overlay = frame.copy()
            cv2.line(overlay, lp1, lp2, _CORRIDOR_COLOR, 3, cv2.LINE_AA)
            cv2.line(overlay, rp1, rp2, _CORRIDOR_COLOR, 3, cv2.LINE_AA)
            frame = cv2.addWeighted(overlay, 0.8, frame, 0.2, 0)

            # Overlay 2: expert bbox crop ghost
            expert_frame_idx = min(
                int((frame_idx / learner_total_frames) * expert_total_frames),
                expert_max_frame,
            )

            # Advance expert cap sequentially until we reach the target frame
            while expert_pos < expert_frame_idx:
                ret_e, next_frame = expert_cap.read()
                if not ret_e:
                    break
                expert_frame = next_frame
                expert_pos += 1

            ret_e = expert_frame is not None

            if ret_e and expert_frame is not None:
                x1 = int(bbox_x1_all[expert_frame_idx])
                y1 = int(bbox_y1_all[expert_frame_idx])
                x2 = int(bbox_x2_all[expert_frame_idx])
                y2 = int(bbox_y2_all[expert_frame_idx])

                # Clamp bbox to expert frame dimensions
                eh, ew = expert_frame.shape[:2]
                x1 = max(0, min(x1, ew - 1))
                y1 = max(0, min(y1, eh - 1))
                x2 = max(x1 + 1, min(x2, ew))
                y2 = max(y1 + 1, min(y2, eh))

                expert_crop = expert_frame[y1:y2, x1:x2]
                crop_h, crop_w = expert_crop.shape[:2]

                paste_x = int(x1 + dx)
                paste_y = int(y1 + dy)

                # Clamp paste region to learner frame dimensions
                px1 = max(0, paste_x)
                py1 = max(0, paste_y)
                px2 = min(width, paste_x + crop_w)
                py2 = min(height, paste_y + crop_h)

                if px2 > px1 and py2 > py1:
                    # Corresponding slice of the crop (in case it was clipped)
                    cx1 = px1 - paste_x
                    cy1 = py1 - paste_y
                    cx2 = cx1 + (px2 - px1)
                    cy2 = cy1 + (py2 - py1)

                    crop_slice = expert_crop[cy1:cy2, cx1:cx2]

                    # tint the crop green
                    green_tint = crop_slice.copy()
                    green_tint[:, :, 0] = (green_tint[:, :, 0] * 0.2).astype(np.uint8)  # reduce blue
                    green_tint[:, :, 1] = np.clip(green_tint[:, :, 1] * 1.5, 0, 255).astype(np.uint8)  # boost green
                    green_tint[:, :, 2] = (green_tint[:, :, 2] * 0.2).astype(np.uint8)  # reduce red

                    # blend green-tinted crop onto learner frame
                    roi = frame[py1:py2, px1:px2]
                    blended = cv2.addWeighted(green_tint, 0.6, roi, 0.4, 0)
                    frame[py1:py2, px1:px2] = blended

            writer.write(frame)
            frame_idx += 1
    finally:
        cap.release()
        expert_cap.release()
        writer.release()

    # Re-encode to H.264 for browser playback
    if _convert_to_web_mp4(tmp_path, web_tmp_path):
        tmp_path.unlink(missing_ok=True)
        web_tmp_path.replace(out_path)
    else:
        tmp_path.replace(out_path)

    return str(out_path)


def _extend_line_to_frame(
    p1: np.ndarray,
    p2: np.ndarray,
    width: int,
    height: int,
) -> tuple[tuple[int, int], tuple[int, int]]:
    """Return (top_pt, bottom_pt) where the line intersects y=0 and y=height."""
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
