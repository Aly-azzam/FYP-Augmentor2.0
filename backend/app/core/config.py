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

    # YOLO + SAM2 scissors tracking
    ROBOFLOW_API_KEY: str | None = None
    ROBOFLOW_MODEL_ID: str = "scissors_ego_real/1"
    ROBOFLOW_CONFIDENCE: float = 0.5
    FRAME_STRIDE: int = 5
    SAM2_MAX_PROCESSED_FRAMES: int | None = None
    SAM2_YOLO_BBOX_SHRINK: float = 0.65
    SAM2_RUN_MODE: str = "subprocess"

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

    # Optional computer-vision provider settings.
    roboflow_api_key: str | None = None
    roboflow_model_id: str | None = None
    roboflow_confidence: float | None = None
    frame_stride: int | None = None

    model_config = SettingsConfigDict(
        env_file=str(Path(__file__).resolve().parents[2] / ".env"),
        env_file_encoding="utf-8",
        extra="allow",
    )


settings = Settings()