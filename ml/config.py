import logging
from pathlib import Path

from dotenv import load_dotenv
from pydantic_settings import BaseSettings

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
    mlflow_tracking_uri: str = "http://localhost:5000"
    model_artifact_path: str = str(PROJECT_ROOT / "model" / "model.pkl")
    random_seed: int = 42

    class Config:
        env_file = str(PROJECT_ROOT / ".env")
        case_sensitive = False
        extra = "ignore"


settings = Settings()