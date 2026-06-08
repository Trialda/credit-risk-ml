import logging

from pydantic_settings import BaseSettings

logger = logging.getLogger(__name__)


class Settings(BaseSettings):
    """Application configuration loaded from environment variables.

    Attributes:
        database_url: PostgreSQL connection string.
        model_path: Path to the serialized model artifact.
    """

    database_url: str
    model_path: str = "/app/model/model.pkl"

    class Config:
        env_file = ".env"
        case_sensitive = False


settings = Settings()