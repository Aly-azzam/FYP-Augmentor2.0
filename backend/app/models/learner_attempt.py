"""Legacy compatibility shim.

Use `app.models.attempt.Attempt` as the canonical ORM model.
"""

from app.models.attempt import Attempt


# Backward-compatible legacy name.
LearnerAttempt = Attempt

__all__ = ["Attempt", "LearnerAttempt"]
