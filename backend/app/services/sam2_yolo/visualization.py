from __future__ import annotations

from pathlib import Path
import subprocess
from typing import Any

import cv2
import numpy as np


def write_sam2_yolo_overlay_video(
    *,
    frame_dir: Path,
    frames: list[dict[str, Any]],
    masks_by_processed_frame: dict[int, np.ndarray],
    output_path: Path,
    fps: float,
    trajectory_metrics: dict[str, Any],
) -> None:
    first_frame = cv2.imread(str(frame_dir / "000000.jpg"))
    if first_frame is None:
        raise RuntimeError(f"Could not read extracted frame from {frame_dir}")

    height, width = first_frame.shape[:2]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_suffix(".tmp.mp4")
    web_tmp_path = output_path.with_suffix(".web.tmp.mp4")
    writer = cv2.VideoWriter(
        str(tmp_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        max(float(fps), 1.0),
        (width, height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Could not create visualization video at {output_path}")

    fitted_line = trajectory_metrics.get("fitted_line")
    drawn_points: list[tuple[int, int]] = []

    try:
        for frame_payload in frames:
            processed_idx = int(frame_payload["processed_frame_index"])
            frame_path = frame_dir / f"{processed_idx:06d}.jpg"
            frame = cv2.imread(str(frame_path))
            if frame is None:
                continue

            mask = masks_by_processed_frame.get(processed_idx)
            if mask is not None:
                frame = _overlay_mask(frame, mask)

            _draw_mask_bbox(frame, frame_payload.get("mask_bbox"))
            _draw_region(frame, frame_payload.get("region"))

            point = frame_payload.get("chosen_tracking_point")
            if point is not None:
                point_tuple = (int(round(point[0])), int(round(point[1])))
                drawn_points.append(point_tuple)
                _draw_trajectory(frame, drawn_points, fitted_line)
                _draw_deviation(frame, point_tuple, fitted_line)

            cv2.putText(
                frame,
                f"SAM2+YOLO frame={frame_payload['frame_index']} point=bbox_center",
                (12, 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            writer.write(frame)
    finally:
        writer.release()

    if output_path.exists():
        output_path.unlink()
    if _convert_to_web_mp4(tmp_path, web_tmp_path):
        tmp_path.unlink(missing_ok=True)
        web_tmp_path.replace(output_path)
    else:
        tmp_path.replace(output_path)


def _convert_to_web_mp4(input_path: Path, output_path: Path) -> bool:
    try:
        import imageio_ffmpeg  # type: ignore
    except ImportError:
        return False

    try:
        ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return False

    command = [
        ffmpeg_exe,
        "-y",
        "-i",
        str(input_path),
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        "-an",
        str(output_path),
    ]
    result = subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    return result.returncode == 0 and output_path.is_file()


def _overlay_mask(frame: np.ndarray, mask: np.ndarray) -> np.ndarray:
    mask_bool = np.asarray(mask, dtype=bool)
    if mask_bool.shape[:2] != frame.shape[:2]:
        mask_bool = cv2.resize(
            mask_bool.astype(np.uint8),
            (frame.shape[1], frame.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        ).astype(bool)

    color = np.zeros_like(frame)
    color[:, :] = (30, 180, 30)
    blended = cv2.addWeighted(frame, 0.65, color, 0.35, 0)
    frame[mask_bool] = blended[mask_bool]
    return frame


def _draw_mask_bbox(frame: np.ndarray, bbox: list[int] | None) -> None:
    if bbox is None:
        return
    x1, y1, x2, y2 = bbox
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)


def _draw_region(frame: np.ndarray, region: dict[str, Any] | None) -> None:
    if not region or region.get("cx") is None or region.get("cy") is None:
        return
    center = (int(round(region["cx"])), int(round(region["cy"])))
    radius = int(round(region["r"]))
    cv2.circle(frame, center, radius, (255, 255, 0), 2)


def _draw_trajectory(
    frame: np.ndarray,
    points: list[tuple[int, int]],
    fitted_line: dict[str, Any] | None,
) -> None:
    if len(points) >= 2:
        cv2.polylines(
            frame,
            [np.array(points, dtype=np.int32)],
            isClosed=False,
            color=(255, 0, 255),
            thickness=2,
        )

    for point in points[-30:]:
        cv2.circle(frame, point, 4, (255, 0, 255), -1)

    current = points[-1]
    cv2.circle(frame, current, 7, (0, 255, 255), -1)
    cv2.putText(
        frame,
        "tracking point",
        (current[0] + 8, max(24, current[1] - 8)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 255, 255),
        2,
        cv2.LINE_AA,
    )

    if fitted_line:
        start = tuple(int(round(value)) for value in fitted_line["start_point"])
        end = tuple(int(round(value)) for value in fitted_line["end_point"])
        cv2.line(frame, start, end, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(
            frame,
            "straight reference",
            (start[0], max(24, start[1] - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )


def _draw_deviation(
    frame: np.ndarray,
    point: tuple[int, int],
    fitted_line: dict[str, Any] | None,
) -> None:
    if not fitted_line:
        return

    line_point = np.array(fitted_line["point"], dtype=np.float32)
    direction = np.array(fitted_line["direction"], dtype=np.float32)
    current = np.array(point, dtype=np.float32)
    projection = line_point + direction * float((current - line_point) @ direction)
    projection_point = tuple(int(round(value)) for value in projection.tolist())
    cv2.line(frame, point, projection_point, (0, 165, 255), 1, cv2.LINE_AA)
