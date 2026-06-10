import json
import logging
from pathlib import Path
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)

SCHEMA_PATH = Path(__file__).resolve().parent / "feature_schema.json"


def load_schema() -> dict[str, Any]:
    """Load the feature schema from disk.

    Returns:
        Parsed feature schema dictionary.
    """
    with open(SCHEMA_PATH) as f:
        return json.load(f)


def validate_feature_dataframe(df: pd.DataFrame) -> None:
    """Validate a feature DataFrame against the feature schema.

    Checks for missing features, incorrect dtypes, and out-of-range
    values. Raises ValueError on the first violation found.

    Args:
        df: Feature DataFrame to validate.

    Raises:
        ValueError: If any schema constraint is violated.
    """
    schema = load_schema()

    numerical = schema["numerical_features"]
    categorical = schema["categorical_features"]
    all_features = {**numerical, **categorical}

    _check_missing_features(df, all_features)
    _check_numerical_ranges(df, numerical)

    logger.info("Feature validation passed")


def _check_missing_features(
    df: pd.DataFrame,
    all_features: dict[str, Any],
) -> None:
    """Check that all required features are present in the DataFrame.

    Args:
        df: Feature DataFrame to validate.
        all_features: Combined numerical and categorical feature specs.

    Raises:
        ValueError: If any required feature is missing.
    """
    missing = set(all_features.keys()) - set(df.columns)
    if missing:
        raise ValueError(
            f"Feature contract violated — missing features: {missing}"
        )


def _check_numerical_ranges(
    df: pd.DataFrame,
    numerical: dict[str, Any],
) -> None:
    """Check that numerical features fall within specified ranges.

    Args:
        df: Feature DataFrame to validate.
        numerical: Numerical feature specs with optional min/max.

    Raises:
        ValueError: If any feature violates its range constraint.
    """
    for feature, spec in numerical.items():
        if feature not in df.columns:
            continue

        col = df[feature].dropna()

        if "min" in spec:
            violations = (col < spec["min"]).sum()
            if violations > 0:
                raise ValueError(
                    f"Feature '{feature}' has {violations} values "
                    f"below minimum {spec['min']}"
                )

        if "max" in spec:
            violations = (col > spec["max"]).sum()
            if violations > 0:
                raise ValueError(
                    f"Feature '{feature}' has {violations} values "
                    f"above maximum {spec['max']}"
                )