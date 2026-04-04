"""Legacy compatibility shim.

Use `app.models.evaluation.Evaluation` as the canonical ORM model.
"""

from app.models.evaluation import Evaluation


# Backward-compatible legacy name.
EvaluationResult = Evaluation

__all__ = ["Evaluation", "EvaluationResult"]
