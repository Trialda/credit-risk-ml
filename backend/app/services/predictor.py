import logging
import pickle
from pathlib import Path
from typing import Optional

import pandas as pd
from sklearn.pipeline import Pipeline

from app.config import settings

logger = logging.getLogger(__name__)

_model: Optional[Pipeline] = None


def load_model() -> None:
    """Load the model artifact from disk into module-level cache.

    Called once at application startup via the lifespan handler.
    Subsequent calls to get_model() return the cached instance.

    Raises:
        RuntimeError: If the model artifact cannot be loaded.
    """
    global _model

    model_path = Path(settings.model_path)

    if not model_path.exists():
        raise RuntimeError(
            f"Model artifact not found at {model_path}. "
            f"Run ml/train.py and ml/export.py first."
        )

    try:
        with open(model_path, "rb") as f:
            _model = pickle.load(f)
        logger.info("Model loaded from %s", model_path)
    except Exception as e:
        raise RuntimeError(f"Failed to load model artifact: {e}") from e


def get_model() -> Pipeline:
    """Return the cached model instance.

    Returns:
        Fitted sklearn Pipeline.

    Raises:
        RuntimeError: If the model has not been loaded yet.
    """
    if _model is None:
        raise RuntimeError(
            "Model is not loaded. Ensure load_model() is called at startup."
        )
    return _model


def predict(features: dict) -> float:
    """Run inference on a single feature dictionary.

    Args:
        features: Dictionary of feature names to values, matching
            the training feature schema.

    Returns:
        Predicted probability of default in range [0, 1].

    Raises:
        ValueError: If the feature input is invalid.
        RuntimeError: If the model is not loaded.
    """
    model = get_model()

    try:
        df = pd.DataFrame([features])
        score = model.predict_proba(df)[:, 1][0]
        return float(score)
    except ValueError as e:
        raise ValueError(f"Invalid feature input: {e}") from e