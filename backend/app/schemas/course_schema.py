from uuid import UUID
from datetime import datetime
from typing import Optional

from pydantic import BaseModel


class ChapterBrief(BaseModel):
    id: UUID
    title: str
    order: int
    has_expert_video: bool

    model_config = {"from_attributes": True}


class ExpertVideoOut(BaseModel):
    id: UUID
    chapter_id: UUID
    file_path: str
    url: str
    duration_seconds: Optional[float] = None
    fps: Optional[float] = None

    model_config = {"from_attributes": True}


class ExpertVideoAssetOut(BaseModel):
    filename: str
    storage_key: str
    file_path: str
    url: str
    chapter_id: Optional[UUID] = None
    expert_video_id: Optional[UUID] = None


class ChapterDetail(BaseModel):
    id: UUID
    course_id: UUID
    title: str
    order: int
    expert_video: Optional[ExpertVideoOut] = None

    model_config = {"from_attributes": True}


class CourseOut(BaseModel):
    id: UUID
    title: str
    description: Optional[str] = None
    instructor: Optional[str] = None
    difficulty: Optional[str] = None
    thumbnail_url: Optional[str] = None
    created_at: datetime

    model_config = {"from_attributes": True}


class CourseDetail(CourseOut):
    chapters: list[ChapterBrief] = []
