"""Evaluation utilities — status transitions, warning builders."""

from app.core.constants import AttemptStatus, ALLOWED_TRANSITIONS


def validate_status_transition(current: AttemptStatus, target: AttemptStatus) -> bool:
    """Check if a status transition is allowed."""
    return target in ALLOWED_TRANSITIONS.get(current, [])


def build_warning(code: str, message: str) -> dict:
    """Build a warning dict matching the Warning schema."""
    return {"code": code, "message": message}
