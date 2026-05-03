from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from app.services.sam2_yolo.visualization import _convert_to_web_mp4


DEFAULT_SMOOTHING_WINDOW = 5
DEFAULT_JUMP_THRESHOLD_PX = 50.0
DEFAULT_RDP_EPSILON_PX = 10.0
DEFAULT_TRAJECTORY_POINT_MODE = "blade_tip_candidate"
VALID_TRAJECTORY_POINT_MODES = {
    "bbox_center",
    "chosen_tracking_point",
    "blade_tip_candidate",
}


def build_cleaned_trajectory_artifact(
    *,
    raw_json_path: Path,
    metrics_json_path: Path,
    output_path: Path,
    trajectory_point_mode: str = DEFAULT_TRAJECTORY_POINT_MODE,
    smoothing_window: int = DEFAULT_SMOOTHING_WINDOW,
    default_jump_threshold_px: float = DEFAULT_JUMP_THRESHOLD_PX,
    rdp_epsilon_px: float = DEFAULT_RDP_EPSILON_PX,
) -> dict[str, Any]:
    if trajectory_point_mode not in VALID_TRAJECTORY_POINT_MODES:
        raise ValueError(
            "trajectory_point_mode must be one of "
            f"{sorted(VALID_TRAJECTORY_POINT_MODES)}; got {trajectory_point_mode!r}"
        )
    raw_payload = _read_json(raw_json_path)
    metrics_payload = _read_json(metrics_json_path) if metrics_json_path.is_file() else {}
    frames = raw_payload.get("frames", [])
    records = [
        _extract_point_record(frame, trajectory_point_mode=trajectory_point_mode)
        for frame in frames
    ]

    raw_step_distances = _step_distances(
        [
            (record["raw_x"], record["raw_y"])
            for record in records
            if record["valid"] and record["raw_x"] is not None and record["raw_y"] is not None
        ]
    )
    jump_threshold = _resolve_jump_threshold(
        metrics_payload=metrics_payload,
        step_distances=raw_step_distances,
        default_jump_threshold_px=default_jump_threshold_px,
    )
    removed_outliers = _mark_outliers(records, jump_threshold)
    interpolation_info = _interpolate_records(records)

    clean_points = [
        (record["clean_x"], record["clean_y"])
        for record in records
        if record["clean_x"] is not None and record["clean_y"] is not None
    ]
    clean_step_distances = _step_distances(clean_points)
    smoothed_points = _moving_average(clean_points, window=smoothing_window)
    _assign_smoothed_points(records, smoothed_points)
    simplified_path = _simplify_records_with_rdp(records, epsilon_px=rdp_epsilon_px)

    valid_points = sum(1 for record in records if record["valid"])
    invalid_points = len(records) - valid_points
    outlier_points = sum(1 for record in records if record["was_outlier"])
    interpolated_points = sum(1 for record in records if record["was_interpolated"])
    max_after = max(clean_step_distances) if clean_step_distances else 0.0
    invalid_ratio = invalid_points / max(len(records), 1)
    usable_for_corridor = bool(
        valid_points >= 20
        and invalid_ratio < 0.2
        and clean_points
        and max_after <= jump_threshold
    )

    payload = {
        "run_id": raw_payload.get("run_id"),
        "source_model": "yolo_sam2",
        "frame_stride": raw_payload.get("frame_stride"),
        "point_type_priority": ["chosen_tracking_point", "bbox_center"],
        "point_type_used": trajectory_point_mode,
        "input_raw_json_path": str(raw_json_path),
        "smoothing_method": "moving_average",
        "smoothing_window": int(smoothing_window),
        "simplification_method": "ramer_douglas_peucker",
        "rdp_epsilon_px": round(float(rdp_epsilon_px), 3),
        "outlier_detection_method": "step_distance_threshold",
        "jump_threshold_px": round(float(jump_threshold), 3),
        "usable_for_corridor": usable_for_corridor,
        "quality": {
            "total_points": len(records),
            "valid_points": valid_points,
            "invalid_points": invalid_points,
            "outlier_points": outlier_points,
            "interpolated_points": interpolated_points,
            "simplified_point_count": len(simplified_path),
            "mean_step_distance_before": _rounded_mean(raw_step_distances),
            "mean_step_distance_after": _rounded_mean(clean_step_distances),
            "max_step_distance_before": round(max(raw_step_distances), 3) if raw_step_distances else 0.0,
            "max_step_distance_after": round(max_after, 3),
        },
        "points": records,
        "simplified_path": simplified_path,
        "removed_outliers": removed_outliers,
        "interpolation_info": interpolation_info,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def write_trajectory_line_overlay_video(
    *,
    frame_dir: Path,
    raw_frames: list[dict[str, Any]],
    cleaned_trajectory: dict[str, Any],
    output_path: Path,
    fps: float,
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
        raise RuntimeError(f"Could not create trajectory overlay video at {output_path}")

    points_by_frame = {
        int(point["frame_index"]): point
        for point in cleaned_trajectory.get("points", [])
    }
    raw_path = [
        (int(round(point["raw_x"])), int(round(point["raw_y"])))
        for point in cleaned_trajectory.get("points", [])
        if point.get("raw_x") is not None and point.get("raw_y") is not None
    ]
    full_path = [
        (int(round(point["smoothed_x"])), int(round(point["smoothed_y"])))
        for point in cleaned_trajectory.get("points", [])
        if point.get("smoothed_x") is not None and point.get("smoothed_y") is not None
    ]
    simplified_path = [
        (int(round(point["x"])), int(round(point["y"])))
        for point in cleaned_trajectory.get("simplified_path", [])
        if point.get("x") is not None and point.get("y") is not None
    ]

    try:
        for frame_payload in raw_frames:
            processed_idx = int(frame_payload["processed_frame_index"])
            frame_path = frame_dir / f"{processed_idx:06d}.jpg"
            frame = cv2.imread(str(frame_path))
            if frame is None:
                continue

            if len(raw_path) >= 2:
                cv2.polylines(
                    frame,
                    [np.array(raw_path, dtype=np.int32)],
                    isClosed=False,
                    color=(160, 160, 160),
                    thickness=1,
                    lineType=cv2.LINE_AA,
                )

            if len(full_path) >= 2:
                cv2.polylines(
                    frame,
                    [np.array(full_path, dtype=np.int32)],
                    isClosed=False,
                    color=(0, 255, 255),
                    thickness=3,
                    lineType=cv2.LINE_AA,
                )

            if len(simplified_path) >= 2:
                cv2.polylines(
                    frame,
                    [np.array(simplified_path, dtype=np.int32)],
                    isClosed=False,
                    color=(0, 220, 255),
                    thickness=5,
                    lineType=cv2.LINE_AA,
                )

            for point in cleaned_trajectory.get("points", []):
                if not point.get("was_outlier"):
                    continue
                if point.get("raw_x") is None or point.get("raw_y") is None:
                    continue
                center = (int(round(point["raw_x"])), int(round(point["raw_y"])))
                cv2.drawMarker(
                    frame,
                    center,
                    (0, 0, 255),
                    markerType=cv2.MARKER_TILTED_CROSS,
                    markerSize=14,
                    thickness=2,
                )

            _draw_named_point(frame, frame_payload.get("bbox_center"), (0, 255, 0), "bbox_center")
            _draw_named_point(
                frame,
                frame_payload.get("blade_tip_candidate"),
                (255, 255, 0),
                "blade_tip",
            )
            point = points_by_frame.get(int(frame_payload["frame_index"]))
            if point and point.get("smoothed_x") is not None and point.get("smoothed_y") is not None:
                current = (int(round(point["smoothed_x"])), int(round(point["smoothed_y"])))
                cv2.circle(frame, current, 7, (0, 255, 255), -1)
                cv2.putText(
                    frame,
                    f"trajectory: {cleaned_trajectory.get('point_type_used')}",
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


def _extract_point_record(
    frame: dict[str, Any],
    *,
    trajectory_point_mode: str,
) -> dict[str, Any]:
    tracking_valid = bool(frame.get("tracking_valid", False))
    point: Any = None
    point_source: str | None = None

    if tracking_valid and trajectory_point_mode == "blade_tip_candidate":
        point = frame.get("blade_tip_candidate")
        point_source = "blade_tip_candidate"
    elif tracking_valid and trajectory_point_mode == "bbox_center":
        point = frame.get("bbox_center")
        point_source = "bbox_center"
    elif tracking_valid:
        point = frame.get("chosen_tracking_point")
        point_source = "chosen_tracking_point"
        if point is None:
            point = frame.get("bbox_center")
            point_source = "bbox_center"

    valid = bool(tracking_valid and _is_xy(point))
    raw_x = round(float(point[0]), 3) if valid else None
    raw_y = round(float(point[1]), 3) if valid else None
    return {
        "frame_index": int(frame["frame_index"]),
        "timestamp_sec": round(float(frame.get("timestamp_sec", 0.0)), 6),
        "raw_x": raw_x,
        "raw_y": raw_y,
        "clean_x": raw_x,
        "clean_y": raw_y,
        "smoothed_x": raw_x,
        "smoothed_y": raw_y,
        "valid": valid,
        "was_outlier": False,
        "was_interpolated": False,
        "point_source": point_source if valid else None,
    }


def _simplify_records_with_rdp(
    records: list[dict[str, Any]],
    *,
    epsilon_px: float,
) -> list[dict[str, Any]]:
    candidates = [
        record
        for record in records
        if record.get("smoothed_x") is not None and record.get("smoothed_y") is not None
    ]
    if not candidates:
        return []

    simplified = _ramer_douglas_peucker(candidates, epsilon_px=float(epsilon_px))
    return [
        {
            "x": round(float(record["smoothed_x"]), 3),
            "y": round(float(record["smoothed_y"]), 3),
            "source_frame_index": int(record["frame_index"]),
            "source_timestamp_sec": round(float(record["timestamp_sec"]), 6),
        }
        for record in simplified
    ]


def _ramer_douglas_peucker(
    records: list[dict[str, Any]],
    *,
    epsilon_px: float,
) -> list[dict[str, Any]]:
    if len(records) <= 2:
        return records

    start = np.array([records[0]["smoothed_x"], records[0]["smoothed_y"]], dtype=np.float64)
    end = np.array([records[-1]["smoothed_x"], records[-1]["smoothed_y"]], dtype=np.float64)
    segment = end - start
    segment_length = float(np.linalg.norm(segment))

    max_distance = -1.0
    max_index = 0
    for index, record in enumerate(records[1:-1], start=1):
        point = np.array([record["smoothed_x"], record["smoothed_y"]], dtype=np.float64)
        if segment_length <= 1e-6:
            distance = float(np.linalg.norm(point - start))
        else:
            offset = point - start
            cross_magnitude = abs(float(segment[0] * offset[1] - segment[1] * offset[0]))
            distance = cross_magnitude / segment_length
        if distance > max_distance:
            max_distance = distance
            max_index = index

    if max_distance > epsilon_px:
        left = _ramer_douglas_peucker(records[: max_index + 1], epsilon_px=epsilon_px)
        right = _ramer_douglas_peucker(records[max_index:], epsilon_px=epsilon_px)
        return left[:-1] + right

    return [records[0], records[-1]]


def _draw_named_point(
    frame: np.ndarray,
    point: Any,
    color: tuple[int, int, int],
    label: str,
) -> None:
    if not _is_xy(point):
        return
    center = (int(round(float(point[0]))), int(round(float(point[1]))))
    cv2.circle(frame, center, 6, color, -1)
    cv2.putText(
        frame,
        label,
        (center[0] + 8, max(20, center[1] - 8)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        color,
        1,
        cv2.LINE_AA,
    )


def _mark_outliers(records: list[dict[str, Any]], jump_threshold: float) -> list[dict[str, Any]]:
    removed: list[dict[str, Any]] = []
    previous: dict[str, Any] | None = None
    for record in records:
        if not record["valid"] or record["raw_x"] is None or record["raw_y"] is None:
            continue
        if previous is not None:
            distance = math.dist(
                (float(previous["raw_x"]), float(previous["raw_y"])),
                (float(record["raw_x"]), float(record["raw_y"])),
            )
            if distance > jump_threshold:
                record["was_outlier"] = True
                removed.append(
                    {
                        "frame_index": record["frame_index"],
                        "timestamp_sec": record["timestamp_sec"],
                        "raw_x": record["raw_x"],
                        "raw_y": record["raw_y"],
                        "reason": "step_distance_above_threshold",
                        "step_distance_px": round(float(distance), 3),
                    }
                )
                continue
        previous = record
    return removed


def _interpolate_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    interpolation_info: list[dict[str, Any]] = []
    good_indexes = [
        index
        for index, record in enumerate(records)
        if record["valid"]
        and not record["was_outlier"]
        and record["raw_x"] is not None
        and record["raw_y"] is not None
    ]
    if not good_indexes:
        return interpolation_info

    for index, record in enumerate(records):
        needs_interpolation = record["was_outlier"] or not record["valid"]
        if not needs_interpolation:
            continue

        previous_indexes = [good_index for good_index in good_indexes if good_index < index]
        next_indexes = [good_index for good_index in good_indexes if good_index > index]
        previous_index = previous_indexes[-1] if previous_indexes else None
        next_index = next_indexes[0] if next_indexes else None

        if previous_index is not None and next_index is not None:
            previous = records[previous_index]
            next_record = records[next_index]
            span = max(next_index - previous_index, 1)
            ratio = (index - previous_index) / span
            clean_x = float(previous["raw_x"]) + (float(next_record["raw_x"]) - float(previous["raw_x"])) * ratio
            clean_y = float(previous["raw_y"]) + (float(next_record["raw_y"]) - float(previous["raw_y"])) * ratio
            method = "linear_between_neighbors"
        elif previous_index is not None:
            previous = records[previous_index]
            clean_x = float(previous["raw_x"])
            clean_y = float(previous["raw_y"])
            method = "nearest_previous_valid"
        elif next_index is not None:
            next_record = records[next_index]
            clean_x = float(next_record["raw_x"])
            clean_y = float(next_record["raw_y"])
            method = "nearest_next_valid"
        else:
            continue

        record["clean_x"] = round(clean_x, 3)
        record["clean_y"] = round(clean_y, 3)
        record["was_interpolated"] = True
        interpolation_info.append(
            {
                "frame_index": record["frame_index"],
                "method": method,
                "previous_valid_frame": (
                    records[previous_index]["frame_index"] if previous_index is not None else None
                ),
                "next_valid_frame": (
                    records[next_index]["frame_index"] if next_index is not None else None
                ),
            }
        )

    return interpolation_info


def _moving_average(points: list[tuple[float, float]], *, window: int) -> list[tuple[float, float]]:
    if not points:
        return []
    window = max(1, int(window))
    half_window = window // 2
    smoothed: list[tuple[float, float]] = []
    for index in range(len(points)):
        start = max(0, index - half_window)
        end = min(len(points), index + half_window + 1)
        subset = points[start:end]
        smoothed.append(
            (
                sum(point[0] for point in subset) / len(subset),
                sum(point[1] for point in subset) / len(subset),
            )
        )
    return smoothed


def _assign_smoothed_points(records: list[dict[str, Any]], smoothed_points: list[tuple[float, float]]) -> None:
    smoothed_index = 0
    for record in records:
        if record["clean_x"] is None or record["clean_y"] is None:
            record["smoothed_x"] = None
            record["smoothed_y"] = None
            continue
        smoothed_x, smoothed_y = smoothed_points[smoothed_index]
        record["smoothed_x"] = round(float(smoothed_x), 3)
        record["smoothed_y"] = round(float(smoothed_y), 3)
        smoothed_index += 1


def _resolve_jump_threshold(
    *,
    metrics_payload: dict[str, Any],
    step_distances: list[float],
    default_jump_threshold_px: float,
) -> float:
    metrics_threshold = (
        metrics_payload.get("trajectory_metrics", {}).get("jump_threshold")
        if isinstance(metrics_payload.get("trajectory_metrics"), dict)
        else None
    )
    if metrics_threshold is not None:
        return float(metrics_threshold)
    if step_distances:
        return float(np.mean(step_distances) + 2.5 * np.std(step_distances))
    return float(default_jump_threshold_px)


def _step_distances(points: list[tuple[float, float]]) -> list[float]:
    return [
        float(math.dist((float(x1), float(y1)), (float(x2), float(y2))))
        for (x1, y1), (x2, y2) in zip(points, points[1:])
    ]


def _rounded_mean(values: list[float]) -> float:
    return round(float(np.mean(values)), 3) if values else 0.0


def _is_xy(point: Any) -> bool:
    if not isinstance(point, list | tuple) or len(point) != 2:
        return False
    return point[0] is not None and point[1] is not None


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))
