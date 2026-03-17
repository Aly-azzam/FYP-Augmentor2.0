from uuid import UUID
from typing import Optional

from pydantic import BaseModel


class UploadResponse(BaseModel):
    attempt_id: UUID
    status: str
    message: str


class EvaluationStartRequest(BaseModel):
    attempt_id: UUID


class EvaluationStartResponse(BaseModel):
    evaluation_id: UUID
    attempt_id: UUID
    status: str
    message: str


class EvaluationStatusResponse(BaseModel):
    evaluation_id: UUID
    attempt_id: UUID
    status: str
    current_stage: Optional[str] = None
    message: Optional[str] = None
