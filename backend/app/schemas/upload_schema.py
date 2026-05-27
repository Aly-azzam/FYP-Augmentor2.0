from uuid import UUID
from typing import Optional

from pydantic import BaseModel


class UploadResponse(BaseModel):
    attempt_id: UUID
    chapter_id: UUID
    expert_video_id: Optional[UUID] = None
    upload_status: str
    original_filename: str
    stored_path: str
    video_url: Optional[str] = None
    message: str


class EvaluationStartRequest(BaseModel):
    attempt_id: UUID


class EvaluationStartResponse(BaseModel):
    evaluation_id: UUID
    attempt_id: UUID
    status: str
    message: str
    score: Optional[int] = None
    gate_status: Optional[str] = None
    gate_reasons: Optional[list[str]] = None
    run_id: Optional[str] = None


class EvaluationStatusResponse(BaseModel):
    evaluation_id: UUID
    attempt_id: UUID
    status: str
    current_stage: Optional[str] = None
    message: Optional[str] = None
