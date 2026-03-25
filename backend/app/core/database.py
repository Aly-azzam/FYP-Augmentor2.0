from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
from sqlalchemy.orm import DeclarativeBase

from app.core.config import settings

engine = create_async_engine(settings.DATABASE_URL, echo=settings.DEBUG)

async_session = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)


class Base(DeclarativeBase):
    pass


async def init_db() -> None:
    """Create database tables for the current MVP schema."""
    # Import models here so SQLAlchemy registers them on Base.metadata before create_all.
    import app.models.chapter  # noqa: F401
    import app.models.course  # noqa: F401
    import app.models.evaluation_result  # noqa: F401
    import app.models.expert_video  # noqa: F401
    import app.models.learner_attempt  # noqa: F401
    import app.models.progress  # noqa: F401
    import app.models.user  # noqa: F401

    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)


async def get_db() -> AsyncSession:
    """FastAPI dependency that yields an async DB session."""
    async with async_session() as session:
        yield session
