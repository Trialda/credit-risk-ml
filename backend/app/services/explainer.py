import logging
from typing import Optional

import numpy as np
import pandas as pd
import shap
from sklearn.pipeline import Pipeline

from app.models.schemas import ShapFeature
from app.services.predictor import get_model

logger = logging.getLogger(__name__)

_explainer: Optional[shap.TreeExplainer] = None


def load_explainer() -> None:
    """Initialise the SHAP TreeExplainer from the loaded model.

    Must be called after load_model() since it accesses the fitted
    LightGBM model from inside the sklearn Pipeline.

    Raises:
        RuntimeError: If the model is not loaded or SHAP init fails.
    """
    global _explainer

    try:
        model = get_model()
        lgbm_model = model.named_steps["model"]
        _explainer = shap.TreeExplainer(lgbm_model)
        logger.info("SHAP TreeExplainer initialised")
    except Exception as e:
        raise RuntimeError(f"Failed to initialise SHAP explainer: {e}") from e


def get_explainer() -> shap.TreeExplainer:
    """Return the cached SHAP explainer instance.

    Returns:
        Initialised TreeExplainer.

    Raises:
        RuntimeError: If the explainer has not been loaded yet.
    """
    if _explainer is None:
        raise RuntimeError(
            "Explainer not initialised. "
            "Ensure load_explainer() is called at startup."
        )
    return _explainer


def explain(features: dict) -> tuple[list[ShapFeature], float]:
    """Compute SHAP values for a single prediction.

    Uses TreeExplainer which is optimised for tree-based models
    and runs in O(TLD) time where T=trees, L=leaves, D=depth.
    Typical latency is 10-50ms for LightGBM models.

    Args:
        features: Dictionary of feature names to values.

    Returns:
        Tuple of (shap_features sorted by absolute impact, base_value).

    Raises:
        RuntimeError: If the explainer is not initialised.
        ValueError: If feature input is invalid.
    """
    model = get_model()
    explainer = get_explainer()

    try:
        df = pd.DataFrame([features])
        df_transformed = model.named_steps["preprocessor"].transform(df)

        shap_values = explainer.shap_values(df_transformed)

        if isinstance(shap_values, list):
            shap_values = shap_values[1]

        base_value = float(explainer.expected_value)
        if isinstance(explainer.expected_value, np.ndarray):
            base_value = float(explainer.expected_value[1])

        feature_names = list(features.keys())
        shap_features = [
            ShapFeature(
                feature=name,
                value=features[name],
                shap_value=float(shap_val),
            )
            for name, shap_val in zip(feature_names, shap_values[0])
        ]

        shap_features.sort(key=lambda x: abs(x.shap_value), reverse=True)

        return shap_features, base_value

    except ValueError as e:
        raise ValueError(f"Invalid feature input for SHAP: {e}") from e