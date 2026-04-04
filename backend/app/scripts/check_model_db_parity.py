from __future__ import annotations

import os
from typing import Any

from sqlalchemy import create_engine, inspect

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

TARGET_TABLES = [
    "users",
    "courses",
    "chapters",
    "videos",
    "attempts",
    "evaluations",
    "evaluation_feedback",
]


def _resolve_database_url() -> str:
    try:
        from app.core.config import settings

        return settings.DATABASE_URL
    except Exception:
        database_url = os.getenv("DATABASE_URL")
        if not database_url:
            raise RuntimeError(
                "DATABASE_URL is not available from settings or environment."
            )
        return database_url


def _column_summary(columns: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {
        str(col["name"]): {
            "type": str(col["type"]),
            "nullable": bool(col["nullable"]),
        }
        for col in columns
    }


def main() -> None:
    engine = create_engine(_resolve_database_url())
    inspector = inspect(engine)

    total_missing = 0
    total_extra = 0
    total_type_mismatches = 0
    total_nullability_mismatches = 0

    print("=== ORM vs Live DB parity report ===")
    for table_name in TARGET_TABLES:
        print(f"\n--- {table_name} ---")
        if table_name not in Base.metadata.tables:
            print("missing table in ORM metadata")
            total_missing += 1
            continue

        db_columns_raw = inspector.get_columns(table_name)
        db_columns = _column_summary(db_columns_raw)
        orm_table = Base.metadata.tables[table_name]
        orm_columns = {
            col.name: {"type": str(col.type), "nullable": bool(col.nullable)}
            for col in orm_table.columns
        }

        db_column_names = list(db_columns.keys())
        orm_column_names = list(orm_columns.keys())

        missing_in_model = sorted(name for name in db_column_names if name not in orm_columns)
        extra_in_model = sorted(name for name in orm_column_names if name not in db_columns)
        total_missing += len(missing_in_model)
        total_extra += len(extra_in_model)

        common = sorted(set(db_column_names) & set(orm_column_names))
        type_mismatches: list[dict[str, str]] = []
        nullability_mismatches: list[dict[str, str]] = []
        for col_name in common:
            db_type = db_columns[col_name]["type"]
            orm_type = orm_columns[col_name]["type"]
            if db_type.lower() != orm_type.lower():
                type_mismatches.append(
                    {
                        "column": col_name,
                        "db_type": db_type,
                        "orm_type": orm_type,
                    }
                )
            db_nullable = db_columns[col_name]["nullable"]
            orm_nullable = orm_columns[col_name]["nullable"]
            if db_nullable != orm_nullable:
                nullability_mismatches.append(
                    {
                        "column": col_name,
                        "db_nullable": str(db_nullable),
                        "orm_nullable": str(orm_nullable),
                    }
                )

        total_type_mismatches += len(type_mismatches)
        total_nullability_mismatches += len(nullability_mismatches)

        print("db_columns:", db_column_names)
        print("orm_columns:", orm_column_names)
        print("missing_in_model:", missing_in_model)
        print("extra_in_model:", extra_in_model)
        if type_mismatches:
            print("type_mismatches:", type_mismatches)
        if nullability_mismatches:
            print("nullability_mismatches:", nullability_mismatches)

    print("\n=== Summary ===")
    print(f"missing_in_model = {total_missing}")
    print(f"extra_in_model = {total_extra}")
    print(f"type_mismatches = {total_type_mismatches}")
    print(f"nullability_mismatches = {total_nullability_mismatches}")
    print("success = True" if (total_missing == 0 and total_extra == 0) else "success = False")


if __name__ == "__main__":
    main()

