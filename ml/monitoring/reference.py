import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

REFERENCE_FILENAME = "training_reference.json"


def save_reference_distribution(
    df: pd.DataFrame,
    output_dir: Path,
    numerical_features: list[str],
    categorical_features: list[str],
) -> Path:
    """Compute and save training distribution statistics to disk.

    Saves per-feature statistics used as the PSI baseline. Numerical
    features get mean, std, and percentiles. Categorical features get
    value counts as proportions.

    Args:
        df: Training feature DataFrame (pre-split, full training set).
        output_dir: Directory to write the reference JSON.
        numerical_features: List of numerical feature names.
        categorical_features: List of categorical feature names.

    Returns:
        Path to the saved reference JSON file.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / REFERENCE_FILENAME

    reference = {
        "n_samples": len(df),
        "numerical": {},
        "categorical": {},
    }

    for feature in numerical_features:
        if feature not in df.columns:
            continue

        # Coerce to numeric — some columns may be object dtype
        # if they passed through Postgres TEXT columns
        col = pd.to_numeric(df[feature], errors="coerce").dropna()

        if len(col) == 0:
            continue

        reference["numerical"][feature] = {
            "mean": float(col.mean()),
            "std": float(col.std()),
            "min": float(col.min()),
            "max": float(col.max()),
            "p5": float(np.percentile(col, 5)),
            "p25": float(np.percentile(col, 25)),
            "p50": float(np.percentile(col, 50)),
            "p75": float(np.percentile(col, 75)),
            "p95": float(np.percentile(col, 95)),
        }

    for feature in categorical_features:
        if feature not in df.columns:
            continue
        col = df[feature].dropna().astype(str)
        if len(col) == 0:
            continue
        value_counts = col.value_counts(normalize=True)
        reference["categorical"][feature] = value_counts.to_dict()

    with open(output_path, "w") as f:
        json.dump(reference, f, indent=2)

    logger.info(
        "Training reference distribution saved to %s "
        "(%d numerical, %d categorical features)",
        output_path,
        len(reference["numerical"]),
        len(reference["categorical"]),
    )

    return output_path


def load_reference_distribution(reference_path: Path) -> dict[str, Any]:
    """Load the training reference distribution from disk.

    Args:
        reference_path: Path to the reference JSON file.

    Returns:
        Reference distribution dictionary.

    Raises:
        FileNotFoundError: If the reference file does not exist.
    """
    if not reference_path.exists():
        raise FileNotFoundError(
            f"Training reference not found at {reference_path}. "
            f"Run ml/train.py first."
        )

    with open(reference_path) as f:
        reference = json.load(f)

    logger.info(
        "Loaded training reference distribution: %d samples, "
        "%d numerical features, %d categorical features",
        reference["n_samples"],
        len(reference["numerical"]),
        len(reference["categorical"]),
    )

    return reference