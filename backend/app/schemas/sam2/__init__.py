"""SAM 2 schemas package.

Pydantic models describing the prompts, per-frame records, and JSON
documents produced by the SAM 2 video-segmentation pipeline.
"""

from app.schemas.sam2.sam2_contract_schema import (
    SAM2RawDocument,
    SAM2RawFrame,
    SAM2SummaryDocument,
    SAM2SummaryWorkingRegion,
)
from app.schemas.sam2.sam2_schema import (
    SAM2AreaStats,
    SAM2BoundingBox,
    SAM2DetectionsDocument,
    SAM2FeaturesDocument,
    SAM2FrameFeatures,
    SAM2FrameMask,
    SAM2InitBox,
    SAM2InitPoint,
    SAM2InitPrompt,
    SAM2RunMeta,
    SAM2WorkingRegion,
)

__all__ = [
    "SAM2AreaStats",
    "SAM2BoundingBox",
    "SAM2DetectionsDocument",
    "SAM2FeaturesDocument",
    "SAM2FrameFeatures",
    "SAM2FrameMask",
    "SAM2InitBox",
    "SAM2InitPoint",
    "SAM2InitPrompt",
    "SAM2RawDocument",
    "SAM2RawFrame",
    "SAM2RunMeta",
    "SAM2SummaryDocument",
    "SAM2SummaryWorkingRegion",
    "SAM2WorkingRegion",
]
