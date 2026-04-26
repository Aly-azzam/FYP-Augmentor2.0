from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
from dotenv import load_dotenv

from app.services.sam2_yolo.schemas import InitialPrompt, YoloScissorsDetection


BACKEND_ROOT = Path(__file__).resolve().parents[3]
load_dotenv(BACKEND_ROOT / ".env")

DEFAULT_ROBOFLOW_MODEL_ID = "scissors_ego_real/1"
DEFAULT_ROBOFLOW_CONFIDENCE = float(os.getenv("ROBOFLOW_CONFIDENCE", "0.5"))
DEFAULT_BBOX_SHRINK = float(os.getenv("SAM2_YOLO_BBOX_SHRINK", "0.65"))


class YoloScissorsDetectionError(RuntimeError):
    """Raised when Roboflow YOLO cannot produce a scissors prompt."""


@dataclass(slots=True)
class FirstUsableFrame:
    frame: Any
    frame_index: int
    timestamp_sec: float
    fps: float
    width: int
    height: int


def read_first_usable_frame(video_path: Path) -> FirstUsableFrame:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_index = 0

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if frame is not None and frame.size > 0:
                return FirstUsableFrame(
                    frame=frame,
                    frame_index=frame_index,
                    timestamp_sec=round(frame_index / max(fps, 1e-6), 6),
                    fps=fps,
                    width=width,
                    height=height,
                )
            frame_index += 1
    finally:
        cap.release()

    raise RuntimeError(f"No usable frames found in video: {video_path}")


def detect_scissors_prompt(
    *,
    video_path: Path,
    confidence_threshold: float = DEFAULT_ROBOFLOW_CONFIDENCE,
    debug_image_path: Path | None = None,
    frame_stride: int = 5,
    max_scanned_frames: int = 30,
    bbox_shrink: float = DEFAULT_BBOX_SHRINK,
) -> InitialPrompt:
    prompt_frame = find_initial_yolo_prompt_frame(
        video_path=video_path,
        confidence_threshold=confidence_threshold,
        frame_stride=frame_stride,
        max_scanned_frames=max_scanned_frames,
    )
    detection = prompt_frame.detection
    raw_bbox = [round(float(value), 3) for value in detection.bbox]
    sam2_prompt_bbox = [
        round(float(value), 3)
        for value in shrink_bbox(
            detection.bbox,
            shrink_factor=bbox_shrink,
        )
    ]
    prompt = InitialPrompt(
        frame_index=prompt_frame.frame_index,
        timestamp_sec=prompt_frame.timestamp_sec,
        point=detection.point,
        bbox=sam2_prompt_bbox,
        raw_yolo_bbox=raw_bbox,
        sam2_prompt_bbox=sam2_prompt_bbox,
        bbox_shrink=round(float(bbox_shrink), 6),
        confidence=round(float(detection.confidence), 6),
        class_name=detection.class_name,
    )

    if debug_image_path is not None:
        save_debug_first_frame(
            frame=prompt_frame.frame,
            detection=detection,
            prompt=prompt,
            output_path=debug_image_path,
        )

    return prompt


@dataclass(slots=True)
class YoloPromptFrame:
    frame: Any
    frame_index: int
    timestamp_sec: float
    detection: YoloScissorsDetection


def find_initial_yolo_prompt_frame(
    *,
    video_path: Path,
    confidence_threshold: float,
    frame_stride: int,
    max_scanned_frames: int,
) -> YoloPromptFrame:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    frame_index = 0
    scanned_frames = 0
    errors: list[str] = []

    try:
        while scanned_frames < max(1, max_scanned_frames):
            ok, frame = cap.read()
            if not ok:
                break

            if frame is None or frame.size == 0:
                frame_index += 1
                continue
            if frame_index % max(1, frame_stride) != 0:
                frame_index += 1
                continue

            try:
                detection = detect_best_scissors_roboflow(
                    frame=frame,
                    confidence_threshold=confidence_threshold,
                )
                return YoloPromptFrame(
                    frame=frame,
                    frame_index=frame_index,
                    timestamp_sec=round(frame_index / max(fps, 1e-6), 6),
                    detection=detection,
                )
            except YoloScissorsDetectionError as exc:
                errors.append(f"frame {frame_index}: {exc}")

            scanned_frames += 1
            frame_index += 1
    finally:
        cap.release()

    detail = "; ".join(errors[-5:]) if errors else "no usable prompt frames were scanned"
    raise YoloScissorsDetectionError(
        "YOLO did not detect scissors in the scanned prompt frames "
        f"(stride={frame_stride}, scanned={scanned_frames}). Last errors: {detail}"
    )


def detect_best_scissors_roboflow(
    *,
    frame: Any,
    confidence_threshold: float,
) -> YoloScissorsDetection:
    from inference_sdk import InferenceHTTPClient  # noqa: PLC0415

    api_key = os.getenv("ROBOFLOW_API_KEY", "").strip()
    model_id = os.getenv("ROBOFLOW_MODEL_ID", DEFAULT_ROBOFLOW_MODEL_ID).strip()
    if not api_key:
        raise YoloScissorsDetectionError("Missing ROBOFLOW_API_KEY in .env")
    if not model_id:
        raise YoloScissorsDetectionError("Missing ROBOFLOW_MODEL_ID in .env")

    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp_file:
        frame_path = Path(tmp_file.name)

    try:
        if not cv2.imwrite(str(frame_path), frame):
            raise RuntimeError("Could not write temp frame for Roboflow inference")

        client = InferenceHTTPClient(
            api_url="https://serverless.roboflow.com",
            api_key=api_key,
        )
        result = client.infer(str(frame_path), model_id=model_id)
    finally:
        if frame_path.exists():
            frame_path.unlink()

    predictions = result.get("predictions", []) if isinstance(result, dict) else []
    if not isinstance(predictions, list) or not predictions:
        raise YoloScissorsDetectionError("YOLO did not detect scissors on the first usable frame")

    candidates: list[YoloScissorsDetection] = []
    scissors_candidates: list[YoloScissorsDetection] = []
    for prediction in predictions:
        class_name = str(prediction.get("class", "")).strip()
        confidence = float(prediction.get("confidence", 0.0))
        if confidence < confidence_threshold:
            continue

        detection = YoloScissorsDetection(
            bbox=_roboflow_xywh_to_xyxy(prediction),
            confidence=confidence,
            class_name=class_name or None,
        )
        candidates.append(detection)
        if "scissor" in class_name.lower():
            scissors_candidates.append(detection)

    pool = scissors_candidates or candidates
    if not pool:
        raise YoloScissorsDetectionError("YOLO did not detect scissors on the first usable frame")

    return max(pool, key=lambda detection: detection.confidence)


def save_debug_first_frame(
    *,
    frame: Any,
    detection: YoloScissorsDetection,
    prompt: InitialPrompt,
    output_path: Path,
) -> None:
    debug_frame = frame.copy()
    raw_x1, raw_y1, raw_x2, raw_y2 = [int(round(value)) for value in detection.bbox]
    prompt_bbox = prompt.sam2_prompt_bbox or prompt.bbox
    sx1, sy1, sx2, sy2 = [int(round(value)) for value in prompt_bbox]
    point_x, point_y = [int(round(value)) for value in prompt.point]

    cv2.rectangle(debug_frame, (raw_x1, raw_y1), (raw_x2, raw_y2), (0, 0, 255), 2)
    cv2.putText(
        debug_frame,
        f"YOLO raw {detection.confidence:.2f}",
        (raw_x1, max(24, raw_y1 - 10)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 0, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.rectangle(debug_frame, (sx1, sy1), (sx2, sy2), (0, 255, 255), 2)
    cv2.putText(
        debug_frame,
        f"SAM2 box shrink={prompt.bbox_shrink or DEFAULT_BBOX_SHRINK:.2f}",
        (sx1, max(48, sy1 - 10)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.circle(debug_frame, (point_x, point_y), 8, (255, 0, 0), -1)
    cv2.putText(
        debug_frame,
        "SAM2 positive point",
        (max(0, point_x - 90), max(24, point_y - 14)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 0, 0),
        2,
        cv2.LINE_AA,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), debug_frame):
        raise RuntimeError(f"Could not write debug image: {output_path}")


def _roboflow_xywh_to_xyxy(prediction: dict[str, Any]) -> list[float]:
    x = float(prediction["x"])
    y = float(prediction["y"])
    width = float(prediction["width"])
    height = float(prediction["height"])
    return [
        x - width / 2.0,
        y - height / 2.0,
        x + width / 2.0,
        y + height / 2.0,
    ]


def shrink_bbox(bbox: list[float], shrink_factor: float = DEFAULT_BBOX_SHRINK) -> list[float]:
    x1, y1, x2, y2 = bbox
    center_x = (x1 + x2) / 2.0
    center_y = (y1 + y2) / 2.0
    width = max(1.0, (x2 - x1) * shrink_factor)
    height = max(1.0, (y2 - y1) * shrink_factor)
    return [
        center_x - width / 2.0,
        center_y - height / 2.0,
        center_x + width / 2.0,
        center_y + height / 2.0,
    ]
