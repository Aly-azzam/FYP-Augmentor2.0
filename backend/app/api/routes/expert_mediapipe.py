"""Admin-only routes for the one-time expert MediaPipe preprocessing flow.

**IMPORTANT — architectural note:**

These endpoints are admin/debug utilities. They are NOT part of the
learner compare / evaluation path and the frontend "compare" flow MUST
NOT call them. The canonical way to preprocess an expert video is the
``app/scripts/process_expert_mediapipe.py`` CLI script.

The compare flow only reads the saved expert outputs (file paths +
summary columns) from the ``videos`` row for the expert reference. It
never triggers MediaPipe expert processing at runtime.
"""

from __future__ import annotations

import logging
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.schemas.expert_mediapipe_schema import (
    ExpertMediaPipeProcessRequest,
    ExpertMediaPipeReference,
)
from app.services.expert_mediapipe_service import (
    ExpertMediaPipeError,
    ExpertNotFoundError,
    register_expert_mediapipe_reference,
)


logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/api/admin/experts",
    tags=["Admin — Expert MediaPipe"],
)


@router.post(
    "/{expert_video_id}/mediapipe/process",
    response_model=ExpertMediaPipeReference,
    summary="[ADMIN] Run MediaPipe preprocessing for an expert video",
    description=(
        "Admin/debug endpoint. Runs the MediaPipe pipeline on an expert "
        "reference video ONCE and persists the outputs under "
        "``storage/expert/mediapipe/{expert_video_id}/``. The frontend "
        "compare flow must NOT call this endpoint."
    ),
)
def process_expert_mediapipe(
    expert_video_id: UUID,
    payload: ExpertMediaPipeProcessRequest | None = None,
    db: Session = Depends(get_db),
) -> ExpertMediaPipeReference:
    effective_payload = payload or ExpertMediaPipeProcessRequest()

    try:
        return register_expert_mediapipe_reference(
            db,
            expert_video_id=expert_video_id,
            chapter_id=effective_payload.chapter_id,
            video_path=effective_payload.video_path,
            overwrite=effective_payload.overwrite,
            max_num_hands=effective_payload.max_num_hands,
            min_detection_confidence=effective_payload.min_detection_confidence,
            min_tracking_confidence=effective_payload.min_tracking_confidence,
        )
    except ExpertNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)
        ) from exc
    except ExpertMediaPipeError as exc:
        logger.warning("Expert MediaPipe preprocessing failed: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)
        ) from exc
    except Exception as exc:  # noqa: BLE001 - controlled 500
        logger.exception("Unexpected expert MediaPipe preprocessing error.")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Expert MediaPipe preprocessing failed: {exc}",
        ) from exc
