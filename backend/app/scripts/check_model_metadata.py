from __future__ import annotations

from app.db.base import Base
from app.models import (  # noqa: F401
    Attempt,
    Chapter,
    Course,
    Evaluation,
    EvaluationFeedback,
    User,
    Video,
)


def main() -> None:
    table_names = sorted(Base.metadata.tables.keys())
    print("Loaded ORM tables:")
    for name in table_names:
        print(f"- {name}")


if __name__ == "__main__":
    main()

