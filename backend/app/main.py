from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.core.config import settings
from app.core.database import init_db
from app.api.routes import courses, chapters, uploads, evaluations, history, progress

app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(courses.router)
app.include_router(chapters.router)
app.include_router(uploads.router)
app.include_router(evaluations.router)
app.include_router(history.router)
app.include_router(progress.router)
app.mount("/storage", StaticFiles(directory=settings.STORAGE_ROOT), name="storage")


@app.on_event("startup")
async def startup_event():
    await init_db()


@app.get("/health", tags=["Health"])
async def health_check():
    return {
        "status": "healthy",
        "version": settings.APP_VERSION,
        "pipeline_version": settings.PIPELINE_VERSION,
    }
