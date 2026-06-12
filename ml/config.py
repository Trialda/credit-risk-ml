import os
import logging
from pathlib import Path

from dotenv import load_dotenv
from pydantic_settings import BaseSettings, SettingsConfigDict

logger = logging.getLogger(__name__)

ML_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = ML_DIR.parent
DATA_DIR = PROJECT_ROOT / "data" / "raw"

load_dotenv(PROJECT_ROOT / ".env")


class Settings(BaseSettings):
    """ML pipeline configuration loaded from environment variables.

    Attributes:
        database_url: PostgreSQL connection string.
        mlflow_tracking_uri: MLflow tracking server URI.
        model_artifact_path: Path to write serialized model artifact.
        random_seed: Global random seed for reproducibility.
    """

    database_url: str
    mlflow_tracking_uri: str = "http://mlflow:5000"
    model_artifact_path: str = str(PROJECT_ROOT / "model" / "model.pkl")
    random_seed: int = 42

    model_config = SettingsConfigDict(
        env_file=str(PROJECT_ROOT / ".env"),
        case_sensitive=False,
        extra="ignore",
        protected_namespaces=("settings_",),
    )

    # def get_database_url(self) -> str:
    #     """Return local database URL if available, else default.

    #     When running locally, DATABASE_URL_LOCAL uses localhost.
    #     When running inside Docker, DATABASE_URL uses db hostname.

    #     Returns:
    #         PostgreSQL connection string appropriate for current context.
    #     """
    #     return os.getenv("DATABASE_URL_LOCAL") or self.database_url
    def get_database_url(self) -> str:
        """Return local database URL if available, else default."""
        load_dotenv(PROJECT_ROOT / ".env")
        local_url = os.getenv("DATABASE_URL_LOCAL")
        logger.info(
            "DATABASE_URL_LOCAL from env: %s",
            local_url or "not set"
        )
        return local_url or self.database_url

settings = Settings()