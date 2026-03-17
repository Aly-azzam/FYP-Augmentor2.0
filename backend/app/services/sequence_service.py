"""Sequence Service — build motion sequences from raw landmarks.

Responsibilities:
- clean and filter landmark sequences
- handle missing frames / interpolation
- produce ordered tracked landmark sequences

Owner: Motion Representation Engine (Person 2)
"""

import json
from pathlib import Path
from typing import Any

from app.core.motion_constants import (
    DEFAULT_TRACKED_LANDMARKS,
    DEFAULT_VISIBILITY_THRESHOLD,
)
from app.schemas.motion_schema import MotionFrame, MotionSequenceOutput
from app.utils.sequence_utils import landmark_to_vector, trim_empty_frames


class SequenceService:
    """Build clean time-ordered motion sequences from landmark JSON."""

    required_top_level_keys = {"video_id", "fps", "frames"}
    required_frame_keys = {"frame_index", "timestamp", "landmarks"}

    def load_landmark_sequence(self, path: str | Path) -> dict[str, Any]:
        """Load raw landmark JSON from storage."""
        json_path = Path(path)
        if not json_path.exists():
            raise FileNotFoundError(f"Landmark JSON not found: {json_path}")

        with json_path.open("r", encoding="utf-8") as file:
            data = json.load(file)

        self._validate_landmark_data(data)
        return data

    def build_motion_sequence(
        self,
        landmark_data: dict[str, Any],
        tracked_points: list[str] | None = None,
    ) -> MotionSequenceOutput:
        """Build a deterministic motion sequence from landmark JSON."""
        self._validate_landmark_data(landmark_data)

        tracked_landmarks = tracked_points or DEFAULT_TRACKED_LANDMARKS
        cleaned_frames: list[MotionFrame] = []

        ordered_frames = sorted(
            landmark_data["frames"],
            key=lambda frame: (frame["frame_index"], frame["timestamp"]),
        )

        for frame in ordered_frames:
            positions = {
                landmark_name: self._extract_position(
                    frame["landmarks"].get(landmark_name),
                )
                for landmark_name in tracked_landmarks
            }

            cleaned_frames.append(
                MotionFrame(
                    frame_index=int(frame["frame_index"]),
                    timestamp=float(frame["timestamp"]),
                    positions=positions,
                )
            )

        trimmed_frames = [
            MotionFrame(**frame)
            for frame in trim_empty_frames(
                [frame.model_dump() for frame in cleaned_frames]
            )
        ]

        return MotionSequenceOutput(
            video_id=str(landmark_data["video_id"]),
            fps=float(landmark_data["fps"]),
            sequence_length=len(trimmed_frames),
            tracked_landmarks=list(tracked_landmarks),
            frames=trimmed_frames,
        )

    def extract_landmark_series(
        self,
        sequence: MotionSequenceOutput,
        landmark_name: str,
    ) -> list[list[float]]:
        """Extract the time-ordered vector series for a single landmark."""
        if landmark_name not in sequence.tracked_landmarks:
            raise ValueError(f"Untracked landmark requested: {landmark_name}")

        return [frame.positions[landmark_name] for frame in sequence.frames]

    def _validate_landmark_data(self, landmark_data: dict[str, Any]) -> None:
        """Validate the minimal structure needed for Step 1."""
        missing_keys = self.required_top_level_keys - landmark_data.keys()
        if missing_keys:
            raise ValueError(
                f"Landmark data missing required top-level keys: {sorted(missing_keys)}"
            )

        if not isinstance(landmark_data["frames"], list):
            raise ValueError("'frames' must be a list")

        for index, frame in enumerate(landmark_data["frames"]):
            missing_frame_keys = self.required_frame_keys - frame.keys()
            if missing_frame_keys:
                raise ValueError(
                    f"Frame at index {index} missing required keys: {sorted(missing_frame_keys)}"
                )

            if not isinstance(frame["landmarks"], dict):
                raise ValueError(f"Frame at index {index} has invalid 'landmarks' value")

    def _extract_position(self, landmark: dict[str, Any] | None) -> list[float]:
        """Convert a landmark dict into a safe position vector."""
        if landmark is None:
            return landmark_to_vector(None)

        visibility = float(landmark.get("visibility", 1.0))
        if visibility < DEFAULT_VISIBILITY_THRESHOLD:
            return landmark_to_vector(None)

        return landmark_to_vector(landmark)
