from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    APP_NAME: str = "AugMentor Backend"
    APP_VERSION: str = "0.1.0"
    DEBUG: bool = True

    DATABASE_URL: str

    # Local storage root (MVP)
    STORAGE_ROOT: Path = Path(__file__).resolve().parents[2] / "storage"
    UPLOAD_DIR: str = "uploads"
    LEARNER_UPLOAD_SUBDIR: str = "learner_videos"
    PROCESSED_DIR: str = "processed"
    OUTPUT_DIR: str = "outputs"
    MAX_UPLOAD_SIZE_BYTES: int = 100 * 1024 * 1024
    STANDARDIZATION_TARGET_FPS: float = 30.0
    STANDARDIZATION_FRAME_WIDTH: int = 640
    STANDARDIZATION_FRAME_HEIGHT: int = 480
    HAND_DETECTION_MAX_NUM_HANDS: int = 2
    HAND_DETECTION_MIN_DETECTION_CONFIDENCE: float = 0.5
    HAND_DETECTION_MIN_TRACKING_CONFIDENCE: float = 0.5
    HAND_DETECTION_MIN_PRESENCE_CONFIDENCE: float = 0.5
    HAND_LANDMARKER_MODEL_PATH: Path = Path(__file__).resolve().parents[2] / "models" / "hand_landmarker.task"

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
    EXPLANATION_MODE: str = "rule"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8"
    )


settings = Settings()