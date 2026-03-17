from pathlib import Path
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    APP_NAME: str = "AugMentor Backend"
    APP_VERSION: str = "0.1.0"
    DEBUG: bool = True

    DATABASE_URL: str = "postgresql+asyncpg://augmentor:augmentor@localhost:5432/augmentor"

    # Local storage root (MVP)
    STORAGE_ROOT: Path = Path(__file__).resolve().parents[2] / "storage"
    UPLOAD_DIR: str = "uploads"
    PROCESSED_DIR: str = "processed"
    OUTPUT_DIR: str = "outputs"

    # Pipeline timeouts (seconds)
    UPLOAD_VALIDATION_TIMEOUT: int = 30
    PERCEPTION_TIMEOUT: int = 120
    MOTION_TIMEOUT: int = 120
    EVALUATION_TIMEOUT: int = 120
    VLM_TIMEOUT: int = 60

    # Versioning
    PIPELINE_VERSION: str = "0.1.0"
    MODEL_VERSION: str = "0.1.0"
    CONFIG_VERSION: str = "0.1.0"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


settings = Settings()
