"""Learner-side scissor-tip tracking constrained by SAM2 hand ROI."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from app.core.sam2_constants import (
    SAM2_FRAMES_SUBDIR,
    SAM2_TIP_ANNOTATED_FILENAME,
    SAM2_TIP_INIT_FILENAME,
    SAM2_TIP_MAX_JUMP_PX,
    SAM2_TIP_MIN_CONFIDENCE,
    SAM2_TIP_REUSE_MAX_FRAMES,
    SAM2_TIP_ROI_MARGIN_PX,
    SAM2_TIP_SMOOTHING_WINDOW,
    SAM2_TIP_TEMPLATE_SIZE_PX,
    SAM2_TIP_TRACKING_FILENAME,
)
from app.schemas.sam2.sam2_contract_schema import SAM2RawDocument
from app.schemas.sam2.tip_tracking_schema import (
    SAM2TipInitialization,
    SAM2TipTrackingDocument,
    SAM2TipTrackingFrame,
)
from app.utils.sam2.sam2_utils import write_video_from_frames


class SAM2TipTrackingError(RuntimeError):
    """Raised when learner-side tip tracking cannot be completed."""


@dataclass
class SAM2TipTrackingArtifacts:
    tip_init_path: Path
    tip_tracking_path: Path
    tip_annotated_video_path: Optional[Path]
    tip_tracking: SAM2TipTrackingDocument


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def _clamp_point(x: float, y: float, width: int, height: int) -> Tuple[float, float]:
    return (
        max(0.0, min(float(width - 1), float(x))),
        max(0.0, min(float(height - 1), float(y))),
    )


def _patch_bbox(center_xy: Sequence[float], side: int, width: int, height: int) -> List[int]:
    half = max(2, int(side // 2))
    cx, cy = float(center_xy[0]), float(center_xy[1])
    x1 = max(0, int(round(cx)) - half)
    y1 = max(0, int(round(cy)) - half)
    x2 = min(max(0, width - 1), int(round(cx)) + half)
    y2 = min(max(0, height - 1), int(round(cy)) + half)
    return [x1, y1, x2, y2]


def _extract_patch(gray: np.ndarray, bbox: Sequence[int]) -> Optional[np.ndarray]:
    x1, y1, x2, y2 = (int(v) for v in bbox)
    if x2 <= x1 or y2 <= y1:
        return None
    patch = gray[y1 : y2 + 1, x1 : x2 + 1]
    if patch.size == 0:
        return None
    return patch


def _moving_average(points: Sequence[Tuple[float, float]], window: int) -> Tuple[float, float]:
    tail = list(points[-max(1, int(window)):])
    xs = np.asarray([p[0] for p in tail], dtype=np.float64)
    ys = np.asarray([p[1] for p in tail], dtype=np.float64)
    return float(xs.mean()), float(ys.mean())


def _search_window_from_roi(
    roi_bbox: Optional[Sequence[int]],
    width: int,
    height: int,
    margin_px: int,
) -> List[int]:
    if roi_bbox is None:
        return [0, 0, max(0, width - 1), max(0, height - 1)]
    x1, y1, x2, y2 = (int(v) for v in roi_bbox)
    return [
        max(0, x1 - int(margin_px)),
        max(0, y1 - int(margin_px)),
        min(max(0, width - 1), x2 + int(margin_px)),
        min(max(0, height - 1), y2 + int(margin_px)),
    ]


def _inside_bbox(point_xy: Sequence[float], bbox: Optional[Sequence[int]]) -> bool:
    if bbox is None:
        return True
    x, y = float(point_xy[0]), float(point_xy[1])
    x1, y1, x2, y2 = (float(v) for v in bbox)
    return x1 <= x <= x2 and y1 <= y <= y2


def _load_frame_map(run_dir: Path, raw: SAM2RawDocument) -> List[Tuple[int, Path, Optional[List[int]]]]:
    frames_dir = run_dir / SAM2_FRAMES_SUBDIR
    if not frames_dir.is_dir():
        raise SAM2TipTrackingError(f"SAM2 frames folder missing: {frames_dir}")
    ordered: List[Tuple[int, Path, Optional[List[int]]]] = []
    for frame in raw.frames:
        frame_idx = int(frame.frame_index)
        frame_path = frames_dir / f"{frame_idx:06d}.jpg"
        if not frame_path.is_file():
            continue
        roi = frame.bbox or frame.mask_bbox_xyxy
        ordered.append((frame_idx, frame_path, roi))
    if not ordered:
        raise SAM2TipTrackingError("No extracted SAM2 frames available for tip tracking.")
    return ordered


def _render_tip_overlay_video(
    *,
    run_dir: Path,
    frame_map: Sequence[Tuple[int, Path, Optional[List[int]]]],
    tracking_frames: Sequence[SAM2TipTrackingFrame],
    fps: float,
) -> Optional[Path]:
    overlay_dir = run_dir / "tip_overlay_frames"
    overlay_dir.mkdir(parents=True, exist_ok=True)
    rendered_paths: List[Path] = []

    tip_by_index: Dict[int, SAM2TipTrackingFrame] = {
        int(item.frame_index): item for item in tracking_frames
    }
    trail: List[Tuple[int, int]] = []

    for frame_index, frame_path, roi_bbox in frame_map:
        image = cv2.imread(str(frame_path))
        if image is None:
            continue

        track = tip_by_index.get(int(frame_index))
        if roi_bbox is not None:
            x1, y1, x2, y2 = (int(v) for v in roi_bbox)
            cv2.rectangle(image, (x1, y1), (x2, y2), (255, 255, 0), 2)

        if track is not None:
            tx, ty = int(round(float(track.tip_point[0]))), int(round(float(track.tip_point[1])))
            trail.append((tx, ty))
            trail = trail[-12:]
            for i, (px, py) in enumerate(trail):
                alpha = (i + 1) / max(1, len(trail))
                color = (0, int(80 + 120 * alpha), 255)
                cv2.circle(image, (px, py), radius=2, color=color, thickness=-1)

            status_color = (0, 255, 255) if track.tracking_status == "ok" else (0, 0, 255)
            cv2.circle(image, (tx, ty), radius=5, color=status_color, thickness=-1)
            cv2.putText(
                image,
                f"Tracked Scissor Tip [{track.tracking_status}]",
                (12, 46),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                status_color,
                2,
                cv2.LINE_AA,
            )

        cv2.putText(
            image,
            "SAM2 Learner Local ROI",
            (12, 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )
        output_frame = overlay_dir / f"tip_overlay_{int(frame_index):06d}.jpg"
        cv2.imwrite(str(output_frame), image)
        rendered_paths.append(output_frame)

    if not rendered_paths:
        return None

    return write_video_from_frames(
        rendered_paths,
        run_dir / SAM2_TIP_ANNOTATED_FILENAME,
        fps=float(fps) if fps > 0 else 30.0,
    )


def track_scissor_tip_in_sam_roi(
    *,
    run_dir: Path,
    raw_document: SAM2RawDocument,
    frame_index: int,
    tip_point_xy: Sequence[float],
) -> SAM2TipTrackingArtifacts:
    """Track a manually-seeded scissor tip through frames constrained by SAM ROI."""
    ordered_frames = _load_frame_map(run_dir, raw_document)
    if frame_index < 0 or frame_index >= len(ordered_frames):
        raise SAM2TipTrackingError(
            f"tip seed frame_index={frame_index} is out of range for processed frames ({len(ordered_frames)})."
        )

    seed_abs_frame_idx, seed_path, _ = ordered_frames[int(frame_index)]
    seed_image = cv2.imread(str(seed_path), cv2.IMREAD_GRAYSCALE)
    if seed_image is None:
        raise SAM2TipTrackingError(f"Could not read seed frame: {seed_path}")
    image_h, image_w = int(seed_image.shape[0]), int(seed_image.shape[1])

    sx, sy = _clamp_point(float(tip_point_xy[0]), float(tip_point_xy[1]), image_w, image_h)
    seed_bbox = _patch_bbox([sx, sy], int(SAM2_TIP_TEMPLATE_SIZE_PX), image_w, image_h)
    template = _extract_patch(seed_image, seed_bbox)
    if template is None:
        raise SAM2TipTrackingError("Could not build tip template from seed click.")

    initialization = SAM2TipInitialization(
        frame_index=int(seed_abs_frame_idx),
        tip_point=[float(sx), float(sy)],
        object="scissor_tip",
        source="manual_click",
    )

    tip_frames: List[SAM2TipTrackingFrame] = []
    warnings: List[str] = []
    accepted_points: List[Tuple[float, float]] = [(float(sx), float(sy))]
    last_stable_point: Tuple[float, float] = (float(sx), float(sy))
    reuse_budget = int(SAM2_TIP_REUSE_MAX_FRAMES)

    for local_idx, (abs_frame_idx, frame_path, roi_bbox) in enumerate(ordered_frames):
        gray = cv2.imread(str(frame_path), cv2.IMREAD_GRAYSCALE)
        if gray is None:
            tip_frames.append(
                SAM2TipTrackingFrame(
                    frame_index=int(abs_frame_idx),
                    tip_point=[float(last_stable_point[0]), float(last_stable_point[1])],
                    tip_bbox=_patch_bbox(last_stable_point, int(SAM2_TIP_TEMPLATE_SIZE_PX), image_w, image_h),
                    tracking_status="missing",
                    confidence=0.0,
                    inside_sam_roi=False,
                )
            )
            continue

        if local_idx < int(frame_index):
            tip_frames.append(
                SAM2TipTrackingFrame(
                    frame_index=int(abs_frame_idx),
                    tip_point=[float(last_stable_point[0]), float(last_stable_point[1])],
                    tip_bbox=_patch_bbox(last_stable_point, int(SAM2_TIP_TEMPLATE_SIZE_PX), image_w, image_h),
                    tracking_status="reused_previous",
                    confidence=0.0,
                    inside_sam_roi=_inside_bbox(last_stable_point, roi_bbox),
                )
            )
            continue

        if local_idx == int(frame_index):
            tip_frames.append(
                SAM2TipTrackingFrame(
                    frame_index=int(abs_frame_idx),
                    tip_point=[float(sx), float(sy)],
                    tip_bbox=seed_bbox,
                    tracking_status="ok",
                    confidence=1.0,
                    inside_sam_roi=_inside_bbox([sx, sy], roi_bbox),
                )
            )
            continue

        search_bbox = _search_window_from_roi(
            roi_bbox,
            width=image_w,
            height=image_h,
            margin_px=int(SAM2_TIP_ROI_MARGIN_PX),
        )
        sx1, sy1, sx2, sy2 = (int(v) for v in search_bbox)
        search_region = gray[sy1 : sy2 + 1, sx1 : sx2 + 1]
        if (
            search_region.size == 0
            or search_region.shape[0] < template.shape[0]
            or search_region.shape[1] < template.shape[1]
        ):
            confidence = 0.0
            proposed = last_stable_point
        else:
            match = cv2.matchTemplate(search_region, template, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, max_loc = cv2.minMaxLoc(match)
            confidence = float(max(0.0, min(1.0, max_val)))
            proposed = (
                float(sx1 + max_loc[0] + (template.shape[1] * 0.5)),
                float(sy1 + max_loc[1] + (template.shape[0] * 0.5)),
            )

        proposed = _clamp_point(proposed[0], proposed[1], image_w, image_h)
        inside_roi = _inside_bbox(proposed, roi_bbox)
        jump = float(
            np.hypot(float(proposed[0]) - float(last_stable_point[0]), float(proposed[1]) - float(last_stable_point[1]))
        )
        unstable = (
            confidence < float(SAM2_TIP_MIN_CONFIDENCE)
            or (not inside_roi)
            or jump > float(SAM2_TIP_MAX_JUMP_PX)
        )

        if unstable:
            status = "unstable"
            if reuse_budget > 0:
                status = "reused_previous"
                reuse_budget -= 1
            proposed = last_stable_point
        else:
            reuse_budget = int(SAM2_TIP_REUSE_MAX_FRAMES)
            accepted_points.append(proposed)
            proposed = _moving_average(accepted_points, int(SAM2_TIP_SMOOTHING_WINDOW))
            last_stable_point = proposed

        tip_frames.append(
            SAM2TipTrackingFrame(
                frame_index=int(abs_frame_idx),
                tip_point=[float(proposed[0]), float(proposed[1])],
                tip_bbox=_patch_bbox(proposed, int(SAM2_TIP_TEMPLATE_SIZE_PX), image_w, image_h),
                tracking_status=status if unstable else "ok",
                confidence=float(confidence),
                inside_sam_roi=_inside_bbox(proposed, roi_bbox),
            )
        )

    tip_document = SAM2TipTrackingDocument(
        run_id=str(raw_document.run_id),
        source_video_path=str(raw_document.source_video_path),
        object="scissor_tip",
        initialization=initialization,
        frames=tip_frames,
        warnings=warnings,
    )

    tip_init_path = Path(run_dir) / SAM2_TIP_INIT_FILENAME
    tip_tracking_path = Path(run_dir) / SAM2_TIP_TRACKING_FILENAME
    _write_json(tip_init_path, initialization.model_dump(mode="json"))
    _write_json(tip_tracking_path, tip_document.model_dump(mode="json"))

    tip_annotated_video_path = _render_tip_overlay_video(
        run_dir=Path(run_dir),
        frame_map=ordered_frames,
        tracking_frames=tip_frames,
        fps=float(raw_document.fps),
    )

    return SAM2TipTrackingArtifacts(
        tip_init_path=tip_init_path,
        tip_tracking_path=tip_tracking_path,
        tip_annotated_video_path=tip_annotated_video_path,
        tip_tracking=tip_document,
    )
