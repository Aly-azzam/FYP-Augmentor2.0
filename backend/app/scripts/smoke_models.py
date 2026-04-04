from __future__ import annotations

from sqlalchemy import text

from app.core.database import SessionLocal
from app.models import Attempt, Chapter, Course, Evaluation, EvaluationFeedback, User, Video


def main() -> None:
    session = SessionLocal()
    checks: dict[str, dict[str, object]] = {}
    ok = True

    try:
        session.execute(text("SELECT 1"))
        checks["select_1"] = {"ok": True}
    except Exception as exc:
        checks["select_1"] = {"ok": False, "error": str(exc)}
        ok = False

    model_checks = [
        ("User", User),
        ("Course", Course),
        ("Chapter", Chapter),
        ("Video", Video),
        ("Attempt", Attempt),
        ("Evaluation", Evaluation),
        ("EvaluationFeedback", EvaluationFeedback),
    ]

    for label, model in model_checks:
        try:
            row = session.query(model).first()
            checks[label] = {"ok": True, "found_row": row is not None}
        except Exception as exc:
            checks[label] = {"ok": False, "error": str(exc)}
            ok = False

    print("=== Smoke query summary ===")
    print({"ok": ok, "checks": checks})
    session.close()


if __name__ == "__main__":
    main()

