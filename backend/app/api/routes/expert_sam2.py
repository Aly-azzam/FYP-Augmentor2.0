"""Admin-only routes for the one-time expert SAM 2 preprocessing flow.

**IMPORTANT — architectural note:**

These endpoints are admin/debug utilities. They are NOT part of the
learner compare / evaluation path and the frontend "compare" flow MUST
NOT call them. The canonical way to preprocess an expert video is the
``app/scripts/process_expert_sam2.py`` CLI script.

The compare flow only reads the saved expert outputs (file paths +
summary columns) from the ``videos`` row for the expert reference. It
never triggers SAM 2 expert processing at runtime.
"""

from __future__ import annotations

import logging
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.schemas.expert_sam2_schema import (
    ExpertSam2ProcessRequest,
    ExpertSam2Reference,
)
from app.services.expert_sam2_service import (
    ExpertSam2AssetsMissingError,
    ExpertMediaPipeMissingError,
    ExpertNotFoundError,
    ExpertSam2Error,
    register_expert_sam2_reference,
)


logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/api/admin/experts",
    tags=["Admin — Expert SAM 2"],
)


@router.post(
    "/{expert_video_id}/sam2/process",
    response_model=ExpertSam2Reference,
    summary="[ADMIN] Run SAM 2 preprocessing for an expert video",
    description=(
        "Admin/debug endpoint. Runs the SAM 2 pipeline on an expert "
        "reference video ONCE and persists the outputs under "
        "``storage/expert/sam2/{expert_video_id}/``. Requires that the "
        "MediaPipe expert reference has already been computed. The "
        "frontend compare flow must NOT call this endpoint."
    ),
)
def process_expert_sam2(
    expert_video_id: UUID,
    payload: ExpertSam2ProcessRequest | None = None,
    db: Session = Depends(get_db),
) -> ExpertSam2Reference:
    effective_payload = payload or ExpertSam2ProcessRequest()

    try:
        return register_expert_sam2_reference(
            db,
            expert_video_id=expert_video_id,
            chapter_id=effective_payload.chapter_id,
            video_path=effective_payload.video_path,
            overwrite=effective_payload.overwrite,
            render_annotation=effective_payload.render_annotation,
        )
    except ExpertNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)
        ) from exc
    except ExpertMediaPipeMissingError as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT, detail=str(exc)
        ) from exc
    except ExpertSam2AssetsMissingError as exc:
        debug_payload = dict(exc.debug)
        debug_payload["frontend_message"] = (
            "SAM2 expert data cannot be generated because required SAM2 model assets are missing."
        )
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={
                "code": "sam2_assets_missing",
                "error_type": "SAM2AssetsMissingError",
                "message": "SAM2 assets are missing. Required checkpoint/config files were not found.",
                "resolved_checkpoint_path": debug_payload.get("resolved_checkpoint_path"),
                "resolved_config_path": debug_payload.get("resolved_config_path"),
                "checkpoint_exists": debug_payload.get("checkpoint_exists"),
                "config_exists": debug_payload.get("config_exists"),
                "debug": debug_payload,
            },
        ) from exc
    except ExpertSam2Error as exc:
        logger.warning("Expert SAM 2 preprocessing failed: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)
        ) from exc
    except Exception as exc:  # noqa: BLE001 - controlled 500
        logger.exception("Unexpected expert SAM 2 preprocessing error.")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Expert SAM 2 preprocessing failed: {exc}",
        ) from exc
