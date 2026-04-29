from __future__ import annotations

from datetime import datetime
from typing import List, Optional

from pydantic import BaseModel, Field


class RunInfo(BaseModel):
    run_id: str = Field(..., description="Unique ID for this optical flow run")
    method: str = Field(default="farneback", description="Optical flow method used")
    created_at: datetime = Field(..., description="Run creation timestamp")
    processing_time_sec: Optional[float] = Field(
        default=None,
        ge=0,
        description="Total processing time in seconds",
    )


class VideoMetadata(BaseModel):
    video_path: str = Field(..., description="Path to the input video")
    fps: float = Field(..., gt=0, description="Frames per second")
    frame_count: int = Field(..., ge=0, description="Total number of frames")
    duration_sec: float = Field(..., ge=0, description="Video duration in seconds")
    width: int = Field(..., gt=0, description="Video width in pixels")
    height: int = Field(..., gt=0, description="Video height in pixels")


class FrameFlowFeatures(BaseModel):
    frame_index: int = Field(..., ge=1, description="Current frame index in the flow pair")
    timestamp_sec: float = Field(..., ge=0, description="Timestamp of the current frame in seconds")

    mean_magnitude: float = Field(..., ge=0, description="Mean optical flow magnitude")
    max_magnitude: float = Field(..., ge=0, description="Maximum optical flow magnitude")
    mean_angle_deg: float = Field(
        ...,
        ge=0,
        le=360,
        description="Mean optical flow angle in degrees",
    )
    motion_area_ratio: float = Field(
        ...,
        ge=0,
        le=1,
        description="Ratio of moving pixels over total pixels",
    )
    roi_used: bool = Field(
        default=False,
        description="True when hand ROI cropping was applied for this frame pair",
    )


class VideoFlowSummary(BaseModel):
    avg_magnitude: float = Field(..., ge=0, description="Average motion magnitude across the video")
    peak_magnitude: float = Field(..., ge=0, description="Peak motion magnitude across the video")
    avg_motion_area_ratio: float = Field(
        ...,
        ge=0,
        le=1,
        description="Average motion area ratio across the video",
    )
    avg_angle_deg: float = Field(
        ...,
        ge=0,
        le=360,
        description="Average motion direction in degrees",
    )
    motion_stability_score: float = Field(
        ...,
        ge=0,
        le=1,
        description="Simple stability score where higher means more stable motion",
    )
    magnitude_std: float = Field(
        ...,
        ge=0,
        description="Standard deviation of per-frame mean motion magnitude",
    )
    magnitude_jitter: float = Field(
        ...,
        ge=0,
        description="Average absolute difference between consecutive mean magnitudes",
    )
    vibration_high_freq_mean: float = Field(
        ...,
        ge=0,
        description="Mean absolute high-frequency residual from smoothed motion magnitude",
    )
    vibration_high_freq_max: float = Field(
        ...,
        ge=0,
        description="Maximum absolute high-frequency residual from smoothed motion magnitude",
    )
    vibration_score: float = Field(
        ...,
        ge=0,
        le=1,
        description="Normalized high-frequency motion instability score",
    )
    roi_enabled: bool = Field(
        default=False,
        description="Whether hand ROI cropping was used for this video summary",
    )
    roi_frames_used: int = Field(
        default=0,
        ge=0,
        description="Number of frame pairs where hand ROI was detected and used",
    )
    roi_fallback_frames: int = Field(
        default=0,
        ge=0,
        description="Number of frame pairs that fell back to full-frame flow",
    )
    roi_usage_ratio: float = Field(
        default=0.0,
        ge=0,
        le=1,
        description="Ratio of frame pairs where hand ROI was used",
    )


class ComparisonInfo(BaseModel):
    frame_count_used: int = Field(..., ge=0, description="Number of frames used in comparison")
    alignment_mode: str = Field(
        default="truncate_to_shorter_video",
        description="How expert and learner sequences were aligned",
    )


class ComparisonMetrics(BaseModel):
    mean_magnitude_difference: float = Field(..., ge=0)
    peak_magnitude_difference: float = Field(..., ge=0)
    motion_area_difference: float = Field(..., ge=0)
    mean_direction_difference_deg: float = Field(..., ge=0, le=180)
    vibration_difference: float = Field(
        default=0.0,
        description="Learner vibration score minus expert vibration score",
    )
    vibration_ratio: float = Field(
        default=0.0,
        ge=0,
        description="Learner vibration score divided by expert vibration score",
    )

    magnitude_curve_mae: float = Field(..., ge=0)
    angle_curve_mae_deg: float = Field(..., ge=0)
    motion_area_curve_mae: float = Field(..., ge=0)


class InterpretationReady(BaseModel):
    main_observation: str = Field(..., description="Short human-readable summary")
    strongest_signal: str = Field(..., description="Most reliable or strongest metric signal")
    weakest_signal: str = Field(..., description="Weakest or least reliable metric signal")


class OpticalFlowSimilarities(BaseModel):
    magnitude_similarity: float = Field(..., ge=0, le=1)
    motion_area_similarity: float = Field(..., ge=0, le=1)
    angle_similarity: float = Field(..., ge=0, le=1)


class OpticalFlowScore(BaseModel):
    optical_flow_score: float = Field(..., ge=0, le=100)


class OpticalFlowEvaluationConfigUsed(BaseModel):
    magnitude_ref: float = Field(..., gt=0)
    motion_area_ref: float = Field(..., gt=0)
    angle_ref_deg: float = Field(..., gt=0)
    magnitude_weight: float = Field(..., ge=0)
    motion_area_weight: float = Field(..., ge=0)
    angle_weight: float = Field(..., ge=0)


class OpticalFlowEvaluationResult(BaseModel):
    similarities: OpticalFlowSimilarities
    score: OpticalFlowScore
    config_used: OpticalFlowEvaluationConfigUsed


class RawOpticalFlowResult(BaseModel):
    run: RunInfo
    expert_video: VideoMetadata
    learner_video: VideoMetadata
    expert_frames: List[FrameFlowFeatures]
    learner_frames: List[FrameFlowFeatures]


class SummaryOpticalFlowResult(BaseModel):
    run: RunInfo
    comparison: ComparisonInfo
    expert_summary: VideoFlowSummary
    learner_summary: VideoFlowSummary
    comparison_metrics: ComparisonMetrics
    interpretation_ready: InterpretationReady
    optical_flow_evaluation: Optional[OpticalFlowEvaluationResult] = None
