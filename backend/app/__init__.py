import sys

# Support launching from project root with `python -m uvicorn backend.app:app`
# while code imports modules via `from app...`.
sys.modules.setdefault("app", sys.modules[__name__])

from .main import app

__all__ = ["app"]
