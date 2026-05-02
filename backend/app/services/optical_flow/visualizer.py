from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from .schemas import FrameFlowFeatures
from .farneback_service import (
    FarnebackConfig,
    _blur_gray_for_flow,
    crop_to_roi,
    embed_roi_flow_in_canvas,
    _effective_roi_source,
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
    frame_features: list[FrameFlowFeatures] | None = None,
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
    active_roi_source = _effective_roi_source(config)
    frame_feature_lookup = (
        {feature.frame_index: feature for feature in frame_features}
        if frame_features is not None
        else None
    )

    if frame_feature_lookup is not None:
        print("[OF] visualization using precomputed ROI metadata", flush=True)
        print("[OF] visualization will NOT rerun YOLO", flush=True)

    if active_roi_source != "none":
        if frame_feature_lookup is not None:
            print("[OF] visualization did not initialize YOLO ROI provider", flush=True)
        elif active_roi_source == "mediapipe_hand":
            from .hand_roi import HandROIDetector

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
        else:
            raise RuntimeError(
                "YOLO Optical Flow visualization requires precomputed frame_features "
                "so visualization does not rerun YOLO."
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
            roi = None
            yolo_debug: dict | None = None
            frame_feature = (
                frame_feature_lookup.get(frame_index)
                if frame_feature_lookup is not None
                else None
            )
            if frame_feature is not None:
                roi_source = frame_feature.roi_source
                active_roi_source = roi_source
                bbox = frame_feature.expanded_roi_xyxy or frame_feature.expanded_roi_bbox
                if bbox is not None and frame_feature.roi_found:
                    roi = tuple(int(value) for value in bbox)
                if roi_source in {
                    "yolo",
                    "yolo_scissors",
                    "yolo_scissors_expanded",
                    "previous_yolo_bbox",
                    "previous_yolo_fallback",
                    "full_frame_fallback",
                }:
                    yolo_debug = {
                        "original_scissors_bbox": frame_feature.scissors_bbox_xyxy,
                        "expanded_roi_bbox_raw": frame_feature.expanded_roi_xyxy,
                        "expanded_roi_bbox_smoothed": frame_feature.expanded_roi_xyxy,
                        "roi_reused_from_previous": frame_feature.roi_reused_from_previous,
                        "fallback_used": frame_feature.fallback_used,
                        "fallback_reason": frame_feature.fallback_reason,
                    }
            elif roi_detector is not None:
                if active_roi_source == "yolo_scissors_expanded":
                    roi = roi_detector.detect(curr_frame, frame_index)
                    yolo_debug = roi_detector.last_debug_dict()
                else:
                    roi = roi_detector.detect(curr_frame)
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
                metric_flow = (
                    roi_flow
                    if active_roi_source
                    in {
                        "yolo",
                        "yolo_scissors",
                        "yolo_scissors_expanded",
                        "previous_yolo_bbox",
                        "previous_yolo_fallback",
                        "full_frame_fallback",
                    }
                    else flow
                )
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
                metric_flow = flow

            vis_frame = flow_to_hsv_bgr(
                flow,
                magnitude_clip_percentile=magnitude_clip_percentile,
            )
            metric_magnitude = np.linalg.norm(metric_flow, axis=2)
            mean_magnitude = (
                float(np.mean(metric_magnitude)) if metric_magnitude.size > 0 else 0.0
            )
            max_magnitude = (
                float(np.max(metric_magnitude)) if metric_magnitude.size > 0 else 0.0
            )
            if frame_feature is not None:
                mean_magnitude = frame_feature.mean_magnitude
                max_magnitude = frame_feature.max_magnitude

            roi_status = ""
            if active_roi_source != "none":
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

                if active_roi_source in {
                    "yolo",
                    "yolo_scissors",
                    "yolo_scissors_expanded",
                    "previous_yolo_bbox",
                    "previous_yolo_fallback",
                    "full_frame_fallback",
                }:
                    if roi_used:
                        reused = (
                            yolo_debug.get("roi_reused_from_previous", False)
                            if yolo_debug is not None
                            else False
                        )
                        roi_status = " [YOLO ROI:reused]" if reused else " [YOLO ROI]"
                    else:
                        roi_status = " [YOLO fallback]"
                elif roi_used:
                    roi_label = (
                        getattr(roi_detector, "last_roi_label", None)
                        if roi_detector is not None
                        else None
                    )
                    roi_status = f" [ROI:{roi_label or 'active'}]"
                else:
                    roi_status = " [fallback]"
                if yolo_debug is not None:
                    raw_bbox = yolo_debug.get("original_scissors_bbox")
                    if raw_bbox is not None:
                        x1, y1, x2, y2 = [int(round(value)) for value in raw_bbox]
                        cv2.rectangle(
                            vis_frame,
                            (x1, y1),
                            (x2, y2),
                            (0, 0, 255),
                            2,
                        )
                        cv2.putText(
                            vis_frame,
                            "YOLO scissors bbox",
                            (x1, max(44, y1 - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.55,
                            (0, 0, 255),
                            2,
                            cv2.LINE_AA,
                        )
                if roi is not None:
                    x1, y1, x2, y2 = roi
                    cv2.rectangle(
                        vis_frame,
                        (x1, y1),
                        (x2, y2),
                        (0, 255, 0),
                        2,
                    )
                    roi_label = (
                        "OF expanded ROI"
                        if active_roi_source
                        in {
                            "yolo",
                            "yolo_scissors",
                            "yolo_scissors_expanded",
                            "previous_yolo_bbox",
                            "previous_yolo_fallback",
                            "full_frame_fallback",
                        }
                        else "Optical Flow ROI"
                    )
                    cv2.putText(
                        vis_frame,
                        roi_label,
                        (x1, max(24, y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.55,
                        (0, 255, 0),
                        2,
                        cv2.LINE_AA,
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
            cv2.putText(
                vis_frame,
                (
                    f"mean_mag={mean_magnitude:.3f} max_mag={max_magnitude:.3f} "
                    f"vibration={(frame_feature.vibration_delta if frame_feature is not None and frame_feature.vibration_delta is not None else 0.0):.3f}"
                ),
                (10, 48),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
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

    print("[OF] visualization finished", flush=True)
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
