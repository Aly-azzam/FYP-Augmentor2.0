"""MediaPipe schemas package.

Pydantic models describing the raw detections, metadata, and derived
features produced by the MediaPipe Hands extraction pipeline.
"""

from app.schemas.mediapipe.mediapipe_schema import (
    MediaPipeDetectionsDocument,
    MediaPipeFeaturesDocument,
    MediaPipeFrameFeatures,
    MediaPipeFrameRaw,
    MediaPipeHandBoundingBox,
    MediaPipeJointAngles,
    MediaPipeLandmarkPoint,
    MediaPipeRunMeta,
    MediaPipeSelectedHandRaw,
)

__all__ = [
    "MediaPipeDetectionsDocument",
    "MediaPipeFeaturesDocument",
    "MediaPipeFrameFeatures",
    "MediaPipeFrameRaw",
    "MediaPipeHandBoundingBox",
    "MediaPipeJointAngles",
    "MediaPipeLandmarkPoint",
    "MediaPipeRunMeta",
    "MediaPipeSelectedHandRaw",
]
