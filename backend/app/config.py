import logging
from pathlib import Path

from pydantic_settings import BaseSettings

logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent.parent


class Settings(BaseSettings):
    """Application configuration loaded from environment variables.

    Attributes:
        database_url: PostgreSQL connection string.
        model_path: Path to the serialized model artifact.
    """

    database_url: str
    model_path: str = "/app/model/model.pkl"

    class Config:
        env_file = str(BASE_DIR / ".env")
        case_sensitive = False
        extra = "ignore"


settings = Settings()