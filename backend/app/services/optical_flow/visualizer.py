from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from .farneback_service import (
    FarnebackConfig,
    _blur_gray_for_flow,
    _resize_frame_if_needed,
    _to_gray,
    _validate_video_path,
)


def ensure_visualization_output_dir(output_dir: str | Path) -> Path:
    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)
    return path


def build_visualization_video_path(
    output_dir: str | Path,
    run_id: str,
    video_role: str,
) -> Path:
    output_path = ensure_visualization_output_dir(output_dir)
    return output_path / f"optical_flow_{run_id}_{video_role}_hsv.mp4"


def flow_to_hsv_bgr(flow: np.ndarray, magnitude_clip_percentile: float = 95.0) -> np.ndarray:
    """
    Convert dense optical flow to an HSV-based BGR visualization.

    Hue   -> direction
    Value -> motion magnitude
    Saturation fixed high for visibility
    """
    if flow.ndim != 3 or flow.shape[2] != 2:
        raise ValueError("Flow must have shape (H, W, 2).")

    flow_x = flow[..., 0]
    flow_y = flow[..., 1]

    magnitude, angle = cv2.cartToPolar(flow_x, flow_y, angleInDegrees=True)

    # OpenCV HSV hue range is [0, 180], while angle is [0, 360]
    hue = angle / 2.0

    if magnitude.size == 0:
        mag_norm = np.zeros_like(magnitude, dtype=np.uint8)
    else:
        clip_value = float(np.percentile(magnitude, magnitude_clip_percentile))
        if clip_value <= 1e-8:
            clip_value = 1.0

        magnitude_clipped = np.clip(magnitude, 0.0, clip_value)
        mag_norm = cv2.normalize(
            magnitude_clipped,
            None,
            0,
            255,
            cv2.NORM_MINMAX,
        ).astype(np.uint8)

    hsv = np.zeros((flow.shape[0], flow.shape[1], 3), dtype=np.uint8)
    hsv[..., 0] = hue.astype(np.uint8)
    hsv[..., 1] = 255
    hsv[..., 2] = mag_norm

    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return bgr


def visualize_video_optical_flow_hsv(
    video_path: str | Path,
    output_video_path: str | Path,
    config: FarnebackConfig | None = None,
    magnitude_clip_percentile: float = 95.0,
    overlay_text: str | None = None,
) -> Path:
    """
    Create an HSV optical flow visualization video for one input video.

    The output video shows motion direction as color and motion strength as brightness.
    """
    config = config or FarnebackConfig()
    input_path = _validate_video_path(video_path)
    output_path = Path(output_video_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {input_path}")

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    fps = fps if fps > 0 else 30.0

    success, prev_frame = cap.read()
    if not success or prev_frame is None:
        cap.release()
        raise RuntimeError(f"Could not read first frame from video: {input_path}")

    prev_frame = _resize_frame_if_needed(
        prev_frame,
        config.resize_width,
        config.resize_height,
    )
    prev_gray = _to_gray(prev_frame)

    height, width = prev_gray.shape[:2]

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"Failed to open VideoWriter for output: {output_path}")

    frame_index = 1
    roi_detector = None

    if config.use_hand_roi:
        from .hand_roi import (
            HandROIDetector,
            crop_to_roi,
            embed_roi_flow_in_canvas,
        )

        effective_roi_padding_px = max(
            int(config.roi_padding_px),
            int(config.roi_enlarge_padding_px),
        )
        roi_detector = HandROIDetector(
            padding_px=effective_roi_padding_px,
            hand_preference=config.roi_hand_preference,
            lock_target=config.roi_lock_target,
            lock_max_missing_frames=config.roi_lock_max_missing_frames,
            lock_max_center_distance_ratio=config.roi_lock_max_center_distance_ratio,
            lock_strict=config.roi_lock_strict,
        )

    try:
        while True:
            success, curr_frame = cap.read()
            if not success or curr_frame is None:
                break

            curr_frame = _resize_frame_if_needed(
                curr_frame,
                config.resize_width,
                config.resize_height,
            )
            curr_gray = _to_gray(curr_frame)

            prev_for_flow = _blur_gray_for_flow(prev_gray, config.gaussian_blur_kernel)
            curr_for_flow = _blur_gray_for_flow(curr_gray, config.gaussian_blur_kernel)
            roi = roi_detector.detect(curr_frame) if roi_detector is not None else None
            roi_used = False

            if roi is not None:
                roi_prev = crop_to_roi(prev_for_flow, roi)
                roi_curr = crop_to_roi(curr_for_flow, roi)
                roi_flow = cv2.calcOpticalFlowFarneback(
                    prev=roi_prev,
                    next=roi_curr,
                    flow=None,
                    pyr_scale=config.pyr_scale,
                    levels=config.levels,
                    winsize=config.winsize,
                    iterations=config.iterations,
                    poly_n=config.poly_n,
                    poly_sigma=config.poly_sigma,
                    flags=config.flags,
                )
                flow = embed_roi_flow_in_canvas(
                    roi_flow=roi_flow,
                    roi=roi,
                    height=height,
                    width=width,
                )
                roi_used = True
            else:
                flow = cv2.calcOpticalFlowFarneback(
                    prev=prev_for_flow,
                    next=curr_for_flow,
                    flow=None,
                    pyr_scale=config.pyr_scale,
                    levels=config.levels,
                    winsize=config.winsize,
                    iterations=config.iterations,
                    poly_n=config.poly_n,
                    poly_sigma=config.poly_sigma,
                    flags=config.flags,
                )

            vis_frame = flow_to_hsv_bgr(
                flow,
                magnitude_clip_percentile=magnitude_clip_percentile,
            )

            roi_status = ""
            if config.use_hand_roi:
                dim_context = cv2.convertScaleAbs(curr_frame, alpha=0.45, beta=12)
                flow_overlay = cv2.addWeighted(
                    dim_context,
                    0.35,
                    vis_frame,
                    0.9,
                    0,
                )
                moving_mask = np.linalg.norm(flow, axis=2) > 1e-6
                vis_frame = np.where(
                    moving_mask[..., None],
                    flow_overlay,
                    dim_context,
                ).astype(np.uint8)

                roi_label = (
                    getattr(roi_detector, "last_roi_label", None)
                    if roi_detector is not None
                    else None
                )
                if roi_used:
                    roi_status = f" [ROI:{roi_label or 'active'}]"
                else:
                    roi_status = " [fallback]"
                if roi is not None:
                    x1, y1, x2, y2 = roi
                    cv2.rectangle(
                        vis_frame,
                        (x1, y1),
                        (x2, y2),
                        (0, 255, 0),
                        2,
                    )

            label = overlay_text if overlay_text else input_path.stem
            cv2.putText(
                vis_frame,
                f"{label}{roi_status} | frame={frame_index}",
                (10, 24),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

            writer.write(vis_frame)

            prev_gray = curr_gray
            frame_index += 1
    finally:
        cap.release()
        writer.release()
        if roi_detector is not None:
            roi_detector.close()

    return output_path


def create_comparison_visualizations(
    expert_video_path: str | Path,
    learner_video_path: str | Path,
    output_dir: str | Path,
    run_id: str,
    config: FarnebackConfig | None = None,
    magnitude_clip_percentile: float = 95.0,
) -> tuple[Path, Path]:
    """
    Create HSV optical flow visualization videos for both expert and learner videos.
    """
    expert_output = build_visualization_video_path(
        output_dir=output_dir,
        run_id=run_id,
        video_role="expert",
    )
    learner_output = build_visualization_video_path(
        output_dir=output_dir,
        run_id=run_id,
        video_role="learner",
    )

    expert_path = visualize_video_optical_flow_hsv(
        video_path=expert_video_path,
        output_video_path=expert_output,
        config=config,
        magnitude_clip_percentile=magnitude_clip_percentile,
        overlay_text="expert",
    )
    learner_path = visualize_video_optical_flow_hsv(
        video_path=learner_video_path,
        output_video_path=learner_output,
        config=config,
        magnitude_clip_percentile=magnitude_clip_percentile,
        overlay_text="learner",
    )

    return expert_path, learner_path
