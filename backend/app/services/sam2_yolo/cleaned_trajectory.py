from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from app.services.sam2_yolo.visualization import _convert_to_web_mp4


DEFAULT_SMOOTHING_WINDOW = 5
DEFAULT_CUTTING_PATH_SMOOTHING_WINDOW = 5
DEFAULT_JUMP_THRESHOLD_PX = 50.0
DEFAULT_OUTLIER_MODE = "local_spike"
DEFAULT_LOCAL_OUTLIER_THRESHOLD_PX = 80.0
DEFAULT_NEIGHBOR_CONSISTENCY_THRESHOLD_PX = 60.0
DEFAULT_RDP_EPSILON_PX = 10.0
DEFAULT_CUTTING_PATH_RDP_EPSILON_PX = 25.0
DEFAULT_TRAJECTORY_POINT_MODE = "blade_tip_candidate"
VALID_TRAJECTORY_POINT_MODES = {
    "bbox_center",
    "chosen_tracking_point",
    "blade_tip_candidate",
}
TELEPORT_WINDOW_RADIUS = 5
TELEPORT_STEP_MULTIPLIER = 3.0
TELEPORT_RETURN_MULTIPLIER = 1.0
ARTIFACT_DETECTION_METHODS = [
    "single_frame_teleport_return",
]


def clean_trajectory(raw_json_path: str, output_path: str) -> dict[str, Any]:
    raw_payload = _read_json(Path(raw_json_path))
    frames = raw_payload.get("frames", [])
    points = _extract_blade_tip_points(frames)
    before_steps = _step_distances(_xy_points(points, x_key="raw_x", y_key="raw_y"))

    removed_single_frame_artifacts = _replace_teleport_return_artifacts(points, before_steps)
    after_steps = _step_distances(_xy_points(points, x_key="clean_x", y_key="clean_y"))
    artifact_points_removed = sum(1 for point in points if point["was_artifact"])
    quality = {
        "total_points": len(points),
        "single_frame_artifacts_removed": len(removed_single_frame_artifacts),
        "artifact_points_removed": artifact_points_removed,
        "interpolated_points": sum(1 for point in points if point["was_interpolated"]),
        "max_step_before": _rounded_max(before_steps),
        "max_step_after": _rounded_max(after_steps),
        "mean_step_before": _rounded_mean(before_steps),
        "mean_step_after": _rounded_mean(after_steps),
    }
    payload = {
        "run_id": raw_payload.get("run_id"),
        "source_model": "yolo_sam2",
        "frame_stride": raw_payload.get("frame_stride"),
        "point_type_used": "blade_tip_candidate",
        "artifact_detection_methods": ARTIFACT_DETECTION_METHODS,
        "teleport_window_radius": TELEPORT_WINDOW_RADIUS,
        "teleport_step_multiplier": TELEPORT_STEP_MULTIPLIER,
        "teleport_return_multiplier": TELEPORT_RETURN_MULTIPLIER,
        "points": points,
        "quality": quality,
        "removed_artifacts": removed_single_frame_artifacts,
        "validation": {
            "top_5_largest_steps_before": _top_step_distances(before_steps),
            "top_5_largest_steps_after": _top_step_distances(after_steps),
            "removed_artifact_frame_indices": _artifact_frame_indices(points),
        },
        "usable_for_corridor": bool(points),
    }

    _print_step_validation(before_steps, after_steps, points, [])

    resolved_output_path = Path(output_path)
    resolved_output_path.parent.mkdir(parents=True, exist_ok=True)
    resolved_output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def smooth_trajectory(
    cleaned_json_path: str,
    output_path: str,
    smoothing_window: int = DEFAULT_CUTTING_PATH_SMOOTHING_WINDOW,
    rdp_epsilon_px: float = DEFAULT_CUTTING_PATH_RDP_EPSILON_PX,
    smoothing_method: str = "moving_average",
) -> dict[str, Any]:
    cleaned_payload = _read_json(Path(cleaned_json_path))
    records = _extract_cleaned_points_for_smoothing(cleaned_payload.get("points", []))
    clean_points = _xy_points(records, x_key="clean_x", y_key="clean_y")
    before_steps = _step_distances(clean_points)

    resolved_window = _resolve_smoothing_window(smoothing_window, len(records))
    smoothed_points, resolved_method = _smooth_xy_points(
        clean_points,
        window=resolved_window,
        method=smoothing_method,
    )
    _assign_cutting_path_smoothed_points(records, smoothed_points)
    after_steps = _step_distances(_xy_points(records, x_key="smoothed_x", y_key="smoothed_y"))
    simplified_path = _simplify_records_with_rdp(records, epsilon_px=rdp_epsilon_px)

    quality = {
        "total_points": len(records),
        "smoothed_points": len(smoothed_points),
        "simplified_points": len(simplified_path),
        "mean_step_before_smoothing": _rounded_mean(before_steps),
        "mean_step_after_smoothing": _rounded_mean(after_steps),
        "max_step_before_smoothing": _rounded_max(before_steps),
        "max_step_after_smoothing": _rounded_max(after_steps),
        "path_length_before_smoothing": round(float(sum(before_steps)), 3),
        "path_length_after_smoothing": round(float(sum(after_steps)), 3),
    }
    payload = {
        "run_id": cleaned_payload.get("run_id"),
        "source_model": "yolo_sam2",
        "point_type_used": "blade_tip_candidate",
        "input_cleaned_trajectory_path": str(cleaned_json_path),
        "smoothing": {
            "method": resolved_method,
            "window": resolved_window,
            "purpose": "remove_scissor_snipping_oscillation",
        },
        "simplification": {
            "method": "ramer_douglas_peucker",
            "rdp_epsilon_px": round(float(rdp_epsilon_px), 3),
            "usage": "visualization_only",
        },
        "corridor_input_points": "smoothed_points",
        "simplified_path_usage": "visualization_only",
        "quality": quality,
        "points": records,
        "simplified_path": simplified_path,
        "usable_for_corridor": bool(records),
    }

    print(f"Top 5 largest steps before smoothing: {_top_step_distances(before_steps)}")
    print(f"Top 5 largest steps after smoothing: {_top_step_distances(after_steps)}")
    print(f"Simplified path points: {len(simplified_path)}")

    resolved_output_path = Path(output_path)
    resolved_output_path.parent.mkdir(parents=True, exist_ok=True)
    resolved_output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def build_cleaned_trajectory_artifact(
    *,
    raw_json_path: Path,
    metrics_json_path: Path,
    output_path: Path,
    trajectory_point_mode: str = DEFAULT_TRAJECTORY_POINT_MODE,
    smoothing_window: int = DEFAULT_SMOOTHING_WINDOW,
    default_jump_threshold_px: float = DEFAULT_JUMP_THRESHOLD_PX,
    outlier_mode: str = DEFAULT_OUTLIER_MODE,
    local_outlier_threshold_px: float = DEFAULT_LOCAL_OUTLIER_THRESHOLD_PX,
    neighbor_consistency_threshold_px: float = DEFAULT_NEIGHBOR_CONSISTENCY_THRESHOLD_PX,
    rdp_epsilon_px: float = DEFAULT_RDP_EPSILON_PX,
) -> dict[str, Any]:
    if trajectory_point_mode == DEFAULT_TRAJECTORY_POINT_MODE:
        return clean_trajectory(str(raw_json_path), str(output_path))

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
    removed_outliers = _mark_outliers(
        records,
        outlier_mode=outlier_mode,
        local_outlier_threshold_px=local_outlier_threshold_px,
        neighbor_consistency_threshold_px=neighbor_consistency_threshold_px,
        debug_jump_threshold_px=jump_threshold,
    )
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
        "outlier_detection_method": outlier_mode,
        "outlier_mode": outlier_mode,
        "local_outlier_threshold_px": round(float(local_outlier_threshold_px), 3),
        "neighbor_consistency_threshold_px": round(float(neighbor_consistency_threshold_px), 3),
        "debug_step_distance_threshold_px": round(float(jump_threshold), 3),
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
        (
            int(round(point.get("smoothed_x", point["clean_x"]))),
            int(round(point.get("smoothed_y", point["clean_y"]))),
        )
        for point in cleaned_trajectory.get("points", [])
        if point.get("clean_x") is not None and point.get("clean_y") is not None
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
                if not (point.get("was_artifact") or point.get("was_outlier")):
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
            if point and point.get("clean_x") is not None and point.get("clean_y") is not None:
                current_x = point.get("smoothed_x", point["clean_x"])
                current_y = point.get("smoothed_y", point["clean_y"])
                current = (int(round(current_x)), int(round(current_y)))
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


def write_smoothed_cutting_path_overlay_video(
    *,
    frame_dir: Path,
    raw_frames: list[dict[str, Any]],
    smoothed_trajectory: dict[str, Any],
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
        raise RuntimeError(f"Could not create smoothed trajectory overlay video at {output_path}")

    trajectory_points = smoothed_trajectory.get("points", [])
    points_by_frame = {
        int(point["frame_index"]): point
        for point in trajectory_points
    }
    raw_frames_by_index = {
        int(frame["frame_index"]): frame
        for frame in raw_frames
    }
    cleaned_path = [
        (int(round(point["clean_x"])), int(round(point["clean_y"])))
        for point in trajectory_points
        if point.get("clean_x") is not None and point.get("clean_y") is not None
    ]
    smoothed_path = [
        (int(round(point["smoothed_x"])), int(round(point["smoothed_y"])))
        for point in trajectory_points
        if point.get("smoothed_x") is not None and point.get("smoothed_y") is not None
    ]
    simplified_path = [
        (int(round(point["x"])), int(round(point["y"])))
        for point in smoothed_trajectory.get("simplified_path", [])
        if point.get("x") is not None and point.get("y") is not None
    ]

    try:
        for frame_payload in raw_frames:
            processed_idx = int(frame_payload["processed_frame_index"])
            frame_path = frame_dir / f"{processed_idx:06d}.jpg"
            frame = cv2.imread(str(frame_path))
            if frame is None:
                continue

            _draw_path(frame, cleaned_path, color=(190, 190, 190), thickness=1)
            _draw_path(frame, smoothed_path, color=(255, 255, 0), thickness=3)
            _draw_path(frame, simplified_path, color=(0, 255, 255), thickness=6)

            for point in trajectory_points:
                if not point.get("was_artifact"):
                    continue
                raw_frame = raw_frames_by_index.get(int(point["frame_index"]), {})
                marker_point = raw_frame.get("blade_tip_candidate") or [
                    point.get("clean_x"),
                    point.get("clean_y"),
                ]
                if not _is_xy(marker_point):
                    continue
                center = (int(round(float(marker_point[0]))), int(round(float(marker_point[1]))))
                cv2.drawMarker(
                    frame,
                    center,
                    (0, 0, 255),
                    markerType=cv2.MARKER_TILTED_CROSS,
                    markerSize=14,
                    thickness=2,
                )

            _draw_named_point(frame, frame_payload.get("bbox_center"), (120, 255, 120), "bbox_center")
            _draw_named_point(
                frame,
                frame_payload.get("blade_tip_candidate"),
                (255, 255, 0),
                "blade_tip",
            )
            current = points_by_frame.get(int(frame_payload["frame_index"]))
            if current and current.get("smoothed_x") is not None and current.get("smoothed_y") is not None:
                center = (int(round(current["smoothed_x"])), int(round(current["smoothed_y"])))
                cv2.circle(frame, center, 7, (255, 255, 0), -1)

            cv2.putText(
                frame,
                "smoothed cutting path",
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


def _extract_blade_tip_points(frames: list[dict[str, Any]]) -> list[dict[str, Any]]:
    points: list[dict[str, Any]] = []
    for frame in frames:
        point = frame.get("blade_tip_candidate")
        if not frame.get("tracking_valid", False) or not _is_xy(point):
            continue
        raw_x = round(float(point[0]), 3)
        raw_y = round(float(point[1]), 3)
        points.append(
            {
                "frame_index": int(frame["frame_index"]),
                "timestamp_sec": round(float(frame.get("timestamp_sec", 0.0)), 6),
                "raw_x": raw_x,
                "raw_y": raw_y,
                "clean_x": raw_x,
                "clean_y": raw_y,
                "was_artifact": False,
                "was_interpolated": False,
                "artifact_reason": None,
            }
        )
    return points


def _extract_cleaned_points_for_smoothing(points: list[dict[str, Any]]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for point in points:
        if point.get("clean_x") is None or point.get("clean_y") is None:
            continue
        records.append(
            {
                "frame_index": int(point["frame_index"]),
                "timestamp_sec": round(float(point.get("timestamp_sec", 0.0)), 6),
                "clean_x": round(float(point["clean_x"]), 3),
                "clean_y": round(float(point["clean_y"]), 3),
                "smoothed_x": round(float(point["clean_x"]), 3),
                "smoothed_y": round(float(point["clean_y"]), 3),
                "was_artifact": bool(point.get("was_artifact", False)),
                "was_interpolated": bool(point.get("was_interpolated", False)),
            }
        )
    return records


def _resolve_smoothing_window(requested_window: int, point_count: int) -> int:
    window = max(1, int(requested_window))
    if window % 2 == 0:
        window += 1
    return window


def _smooth_xy_points(
    points: list[tuple[float, float]],
    *,
    window: int,
    method: str,
) -> tuple[list[tuple[float, float]], str]:
    if not points:
        return [], "moving_average"
    if method == "savitzky_golay":
        smoothed = _savitzky_golay_points(points, window=window)
        if smoothed is not None:
            return smoothed, "savitzky_golay"
    return _moving_average(points, window=window), "moving_average"


def _savitzky_golay_points(
    points: list[tuple[float, float]],
    *,
    window: int,
) -> list[tuple[float, float]] | None:
    if window < 5 or len(points) < window:
        return None
    try:
        from scipy.signal import savgol_filter  # type: ignore
    except ImportError:
        return None

    coordinates = np.asarray(points, dtype=np.float64)
    smoothed_x = savgol_filter(coordinates[:, 0], window_length=window, polyorder=2, mode="interp")
    smoothed_y = savgol_filter(coordinates[:, 1], window_length=window, polyorder=2, mode="interp")
    return [
        (float(x), float(y))
        for x, y in zip(smoothed_x, smoothed_y, strict=True)
    ]


def _assign_cutting_path_smoothed_points(
    records: list[dict[str, Any]],
    smoothed_points: list[tuple[float, float]],
) -> None:
    for record, smoothed_point in zip(records, smoothed_points, strict=True):
        record["smoothed_x"] = round(float(smoothed_point[0]), 3)
        record["smoothed_y"] = round(float(smoothed_point[1]), 3)


def _replace_teleport_return_artifacts(
    points: list[dict[str, Any]],
    step_distances: list[float],
) -> list[dict[str, Any]]:
    removed: list[dict[str, Any]] = []
    if len(points) < 3 or not step_distances:
        return removed

    artifact_indexes: set[int] = set()
    for index in range(1, len(points) - 1):
        local_median = _local_step_median(step_distances, point_index=index)
        if local_median <= 1e-6:
            continue

        previous = points[index - 1]
        current = points[index]
        next_point = points[index + 1]
        step_ab = _record_distance(previous, current, x_key="raw_x", y_key="raw_y")
        step_bc = _record_distance(current, next_point, x_key="raw_x", y_key="raw_y")
        roundtrip = _record_distance(previous, next_point, x_key="raw_x", y_key="raw_y")

        if not (
            step_ab > TELEPORT_STEP_MULTIPLIER * local_median
            and step_bc > TELEPORT_STEP_MULTIPLIER * local_median
            and roundtrip < TELEPORT_RETURN_MULTIPLIER * local_median
        ):
            continue

        artifact_indexes.add(index)
        removed.append(
            {
                "frame_index": current["frame_index"],
                "timestamp_sec": current["timestamp_sec"],
                "raw_x": current["raw_x"],
                "raw_y": current["raw_y"],
                "reason": "single_frame_teleport_return",
                "step_ab": round(float(step_ab), 3),
                "step_bc": round(float(step_bc), 3),
                "roundtrip": round(float(roundtrip), 3),
                "local_median": round(float(local_median), 3),
            }
        )

    for index in artifact_indexes:
        previous = points[index - 1]
        current = points[index]
        next_point = points[index + 1]
        current["clean_x"] = round((float(previous["raw_x"]) + float(next_point["raw_x"])) / 2.0, 3)
        current["clean_y"] = round((float(previous["raw_y"]) + float(next_point["raw_y"])) / 2.0, 3)
        current["was_artifact"] = True
        current["was_interpolated"] = True
        current["artifact_reason"] = "single_frame_teleport_return"

    return removed


def _replace_multi_frame_drift_artifacts(
    points: list[dict[str, Any]],
    step_distances: list[float],
) -> list[dict[str, Any]]:
    removed_segments: list[dict[str, Any]] = []
    if len(points) < MULTI_DRIFT_MIN_POINTS + 2 or not step_distances:
        return removed_segments

    local_medians = [
        _local_step_median(step_distances, point_index=step_index)
        for step_index in range(len(step_distances))
    ]
    abnormal_steps = [
        local_median > 1e-6 and step > MULTI_DRIFT_STEP_MULTIPLIER * local_median
        for step, local_median in zip(step_distances, local_medians, strict=True)
    ]

    for run_start, run_end in _true_runs(abnormal_steps):
        start_index = run_start + 1
        end_index = run_end
        point_count = end_index - start_index + 1
        if point_count < MULTI_DRIFT_MIN_POINTS or point_count > MULTI_DRIFT_MAX_POINTS:
            continue
        if start_index <= 0 or end_index >= len(points) - 1:
            continue
        if points[start_index - 1].get("was_artifact") or points[end_index + 1].get("was_artifact"):
            continue
        if any(point.get("was_artifact") for point in points[start_index : end_index + 1]):
            continue

        segment_step_indexes = list(range(run_start, run_end + 1))
        segment_steps = [step_distances[step_index] for step_index in segment_step_indexes]
        segment_local_medians = [local_medians[step_index] for step_index in segment_step_indexes]
        segment_path_length = float(sum(segment_steps))
        segment_net_displacement = _record_distance(
            points[start_index - 1],
            points[end_index + 1],
            x_key="raw_x",
            y_key="raw_y",
        )
        expected_local_path_length = float(sum(segment_local_medians))
        if expected_local_path_length <= 1e-6:
            continue
        if not (
            segment_path_length > MULTI_DRIFT_PATH_TO_NET_RATIO * max(segment_net_displacement, 1e-6)
            and segment_path_length > MULTI_DRIFT_EXPECTED_PATH_MULTIPLIER * expected_local_path_length
        ):
            continue

        segment_info = {
            "start_frame_index": points[start_index]["frame_index"],
            "end_frame_index": points[end_index]["frame_index"],
            "point_count": point_count,
            "reason": "multi_frame_drift",
            "path_length": round(segment_path_length, 3),
            "net_displacement": round(float(segment_net_displacement), 3),
            "expected_local_path_length": round(expected_local_path_length, 3),
            "artifact_frame_indices": [
                int(point["frame_index"])
                for point in points[start_index : end_index + 1]
            ],
        }
        _interpolate_artifact_segment(
            points,
            start_index=start_index,
            end_index=end_index,
            reason="multi_frame_drift",
        )
        removed_segments.append(segment_info)

    return removed_segments


def _interpolate_artifact_segment(
    points: list[dict[str, Any]],
    *,
    start_index: int,
    end_index: int,
    reason: str,
) -> None:
    previous = points[start_index - 1]
    next_point = points[end_index + 1]
    span = end_index - start_index + 2
    for index in range(start_index, end_index + 1):
        ratio = (index - start_index + 1) / span
        point = points[index]
        point["clean_x"] = round(
            float(previous["raw_x"]) + (float(next_point["raw_x"]) - float(previous["raw_x"])) * ratio,
            3,
        )
        point["clean_y"] = round(
            float(previous["raw_y"]) + (float(next_point["raw_y"]) - float(previous["raw_y"])) * ratio,
            3,
        )
        point["was_artifact"] = True
        point["was_interpolated"] = True
        point["artifact_reason"] = reason


def _true_runs(values: list[bool]) -> list[tuple[int, int]]:
    runs: list[tuple[int, int]] = []
    start: int | None = None
    for index, value in enumerate(values):
        if value and start is None:
            start = index
        elif not value and start is not None:
            runs.append((start, index - 1))
            start = None
    if start is not None:
        runs.append((start, len(values) - 1))
    return runs


def _local_step_median(step_distances: list[float], *, point_index: int) -> float:
    start = max(0, point_index - TELEPORT_WINDOW_RADIUS)
    end = min(len(step_distances), point_index + TELEPORT_WINDOW_RADIUS + 1)
    local_steps = step_distances[start:end]
    if not local_steps:
        return 0.0
    sorted_steps = np.sort(np.asarray(local_steps, dtype=np.float64))
    return float(sorted_steps[(len(sorted_steps) - 1) // 2])


def _xy_points(
    points: list[dict[str, Any]],
    *,
    x_key: str,
    y_key: str,
) -> list[tuple[float, float]]:
    return [
        (float(point[x_key]), float(point[y_key]))
        for point in points
        if point.get(x_key) is not None and point.get(y_key) is not None
    ]


def _record_distance(
    first: dict[str, Any],
    second: dict[str, Any],
    *,
    x_key: str,
    y_key: str,
) -> float:
    first_xy = np.asarray([float(first[x_key]), float(first[y_key])], dtype=np.float64)
    second_xy = np.asarray([float(second[x_key]), float(second[y_key])], dtype=np.float64)
    return float(np.linalg.norm(second_xy - first_xy))


def _top_step_distances(step_distances: list[float], limit: int = 5) -> list[float]:
    return [
        round(float(step), 3)
        for step in sorted(step_distances, reverse=True)[:limit]
    ]


def _artifact_frame_indices(points: list[dict[str, Any]]) -> list[int]:
    return [
        int(point["frame_index"])
        for point in points
        if point.get("was_artifact")
    ]


def _drift_segment_frame_ranges(removed_drift_segments: list[dict[str, Any]]) -> list[list[int]]:
    return [
        [
            int(segment["start_frame_index"]),
            int(segment["end_frame_index"]),
        ]
        for segment in removed_drift_segments
    ]


def _print_step_validation(
    before_steps: list[float],
    after_steps: list[float],
    points: list[dict[str, Any]],
    removed_drift_segments: list[dict[str, Any]],
) -> None:
    print(f"Top 5 largest steps before: {_top_step_distances(before_steps)}")
    print(f"Top 5 largest steps after: {_top_step_distances(after_steps)}")
    print(f"Removed artifact frame indices: {_artifact_frame_indices(points)}")
    print(f"Removed drift segment frame ranges: {_drift_segment_frame_ranges(removed_drift_segments)}")
    mean_after = float(np.mean(after_steps)) if after_steps else 0.0
    max_after = max(after_steps) if after_steps else 0.0
    if mean_after > 0.0 and max_after > 3.0 * mean_after:
        print("Possible remaining trajectory artifacts")


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


def _draw_path(
    frame: np.ndarray,
    points: list[tuple[int, int]],
    *,
    color: tuple[int, int, int],
    thickness: int,
) -> None:
    if len(points) < 2:
        return
    cv2.polylines(
        frame,
        [np.array(points, dtype=np.int32)],
        isClosed=False,
        color=color,
        thickness=thickness,
        lineType=cv2.LINE_AA,
    )


def _mark_outliers(
    records: list[dict[str, Any]],
    *,
    outlier_mode: str,
    local_outlier_threshold_px: float,
    neighbor_consistency_threshold_px: float,
    debug_jump_threshold_px: float,
) -> list[dict[str, Any]]:
    removed: list[dict[str, Any]] = []
    valid_indexes = [
        index
        for index, record in enumerate(records)
        if record["valid"] and record["raw_x"] is not None and record["raw_y"] is not None
    ]

    previous_record: dict[str, Any] | None = None
    for index in valid_indexes:
        record = records[index]
        if previous_record is not None:
            record["debug_step_distance_from_previous_px"] = round(
                _point_distance(previous_record, record),
                3,
            )
        previous_record = record

    if outlier_mode != "local_spike":
        raise ValueError(f"Unsupported outlier_mode: {outlier_mode!r}")

    for valid_position, index in enumerate(valid_indexes):
        if valid_position == 0 or valid_position == len(valid_indexes) - 1:
            continue
        previous = records[valid_indexes[valid_position - 1]]
        record = records[index]
        next_record = records[valid_indexes[valid_position + 1]]
        if not record["valid"] or record["raw_x"] is None or record["raw_y"] is None:
            continue

        previous_to_current = _point_distance(previous, record)
        current_to_next = _point_distance(record, next_record)
        neighbor_distance = _point_distance(previous, next_record)
        expected_x = (float(previous["raw_x"]) + float(next_record["raw_x"])) / 2.0
        expected_y = (float(previous["raw_y"]) + float(next_record["raw_y"])) / 2.0
        local_error = math.dist((float(record["raw_x"]), float(record["raw_y"])), (expected_x, expected_y))
        spike_cosine = _turn_cosine(previous, record, next_record)
        is_sharp_one_frame_spike = (
            previous_to_current > local_outlier_threshold_px
            and current_to_next > local_outlier_threshold_px
            and spike_cosine < -0.35
        )
        neighbors_are_consistent = neighbor_distance <= neighbor_consistency_threshold_px

        record["local_error_px"] = round(float(local_error), 3)
        record["neighbor_consistency_distance_px"] = round(float(neighbor_distance), 3)
        record["local_spike_cosine"] = round(float(spike_cosine), 3)

        if not (
            local_error > local_outlier_threshold_px
            and is_sharp_one_frame_spike
            and neighbors_are_consistent
        ):
            continue

        record["was_outlier"] = True
        removed.append(
            {
                "frame_index": record["frame_index"],
                "timestamp_sec": record["timestamp_sec"],
                "raw_x": record["raw_x"],
                "raw_y": record["raw_y"],
                "reason": "local_one_frame_spike",
                "local_error_px": round(float(local_error), 3),
                "previous_step_distance_px": round(float(previous_to_current), 3),
                "next_step_distance_px": round(float(current_to_next), 3),
                "neighbor_consistency_distance_px": round(float(neighbor_distance), 3),
                "debug_step_distance_threshold_px": round(float(debug_jump_threshold_px), 3),
            }
        )
    return removed


def _point_distance(first: dict[str, Any], second: dict[str, Any]) -> float:
    return float(
        math.dist(
            (float(first["raw_x"]), float(first["raw_y"])),
            (float(second["raw_x"]), float(second["raw_y"])),
        )
    )


def _turn_cosine(
    previous: dict[str, Any],
    current: dict[str, Any],
    next_record: dict[str, Any],
) -> float:
    incoming = np.array(
        [
            float(current["raw_x"]) - float(previous["raw_x"]),
            float(current["raw_y"]) - float(previous["raw_y"]),
        ],
        dtype=np.float64,
    )
    outgoing = np.array(
        [
            float(next_record["raw_x"]) - float(current["raw_x"]),
            float(next_record["raw_y"]) - float(current["raw_y"]),
        ],
        dtype=np.float64,
    )
    denominator = float(np.linalg.norm(incoming) * np.linalg.norm(outgoing))
    if denominator <= 1e-6:
        return 1.0
    return float(np.dot(incoming, outgoing) / denominator)


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
        float(np.linalg.norm(np.asarray([x2 - x1, y2 - y1], dtype=np.float64)))
        for (x1, y1), (x2, y2) in zip(points, points[1:])
    ]


def _rounded_max(values: list[float]) -> float:
    return round(float(max(values)), 3) if values else 0.0


def _rounded_mean(values: list[float]) -> float:
    return round(float(np.mean(values)), 3) if values else 0.0


def _is_xy(point: Any) -> bool:
    if not isinstance(point, list | tuple) or len(point) != 2:
        return False
    return point[0] is not None and point[1] is not None


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    backend_root = Path(__file__).resolve().parents[3]
    runs_root = backend_root / "storage" / "outputs" / "sam2_yolo" / "runs"
    raw_candidates = sorted(
        runs_root.glob("*/raw.json"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if not raw_candidates:
        print(f"No raw.json files found under {runs_root}")
    else:
        raw_path = raw_candidates[0]
        output_path = raw_path.with_name("cleaned_trajectory.json")
        result = clean_trajectory(str(raw_path), str(output_path))
        print(json.dumps(result["quality"], indent=2))
