"""Legacy compatibility shim.

Use `app.models.video.Video` as the canonical ORM model.
"""

from app.models.video import Video


# Backward-compatible legacy name.
ExpertVideo = Video

__all__ = ["Video", "ExpertVideo"]
