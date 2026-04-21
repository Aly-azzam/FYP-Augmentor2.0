"""Global pytest safety guards for database isolation.

This file is intentionally at ``backend/conftest.py`` so it applies to
both ``backend/tests`` and ``backend/app/tests``.

Hard rules:
1. pytest must use ``TEST_DATABASE_URL``.
2. ``TEST_DATABASE_URL`` must not equal the app ``DATABASE_URL``.
3. test DB URL must look like an isolated test database.
"""

from __future__ import annotations

import os


def _normalize_url(url: str) -> str:
    return url.strip().strip('"').strip("'")


def _looks_like_test_database(url: str) -> bool:
    lowered = url.lower()
    return any(
        token in lowered
        for token in (
            "test",
            "pytest",
            "sqlite",
            "tmp",
        )
    )


def _configure_test_database_url() -> str:
    test_database_url = os.getenv("TEST_DATABASE_URL")
    app_database_url = os.getenv("DATABASE_URL")

    if not test_database_url:
        raise RuntimeError(
            "Refusing to run pytest without TEST_DATABASE_URL. "
            "Set TEST_DATABASE_URL to an isolated test database first."
        )

    normalized_test = _normalize_url(test_database_url)
    normalized_app = _normalize_url(app_database_url) if app_database_url else None

    if normalized_app and normalized_test == normalized_app:
        raise RuntimeError(
            "Unsafe test configuration: TEST_DATABASE_URL matches DATABASE_URL. "
            "Point TEST_DATABASE_URL to a separate test database."
        )

    if not _looks_like_test_database(normalized_test):
        raise RuntimeError(
            "Refusing to run pytest because TEST_DATABASE_URL does not look "
            "like a test database URL. Include an explicit test marker such "
            "as 'test'/'pytest' in the DB name."
        )

    # Force all app imports inside pytest to use the test DB URL.
    os.environ["DATABASE_URL"] = normalized_test
    return normalized_test


_EFFECTIVE_TEST_DATABASE_URL = _configure_test_database_url()


def pytest_sessionstart(session) -> None:  # noqa: ARG001
    """Final safety check once pytest starts.

    We update the already-instantiated app settings object so any module
    using ``settings.DATABASE_URL`` points at the test DB, then ensure the
    SQLAlchemy engine in ``app.core.database`` is bound to that URL.
    """
    from sqlalchemy import create_engine

    from app.core.config import settings
    import app.core.database as database_module

    settings.DATABASE_URL = _EFFECTIVE_TEST_DATABASE_URL

    active_engine_url = _normalize_url(str(database_module.engine.url))
    if active_engine_url != _EFFECTIVE_TEST_DATABASE_URL:
        database_module.engine.dispose()
        database_module.engine = create_engine(
            settings.DATABASE_URL,
            echo=settings.DEBUG,
        )
        database_module.SessionLocal.configure(bind=database_module.engine)
