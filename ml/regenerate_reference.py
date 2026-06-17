"""Regenerate training_reference.json without rerunning the full pipeline.

Loads the existing feature table from Postgres (already built), applies
the same train/validation split logic as train.py, and saves the
reference distribution in the current format. Use this when only the
reference file format changed, not the model or feature table.

Run inside the ml container:
    docker compose --profile training run --remove-orphans ml \
        python -m ml.regenerate_reference
"""
import logging
from pathlib import Path

from ml.config import settings
from ml.pipeline.features import build_feature_table
from ml.pipeline.preprocess import (
    CATEGORICAL_FEATURES,
    NUMERICAL_FEATURES,
    prepare_data,
)
from ml.monitoring.reference import save_reference_distribution

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main() -> None:
    """Reload cached feature table and regenerate the reference file."""
    logger.info("Loading existing feature table (skip_if_exists=True)")
    features = build_feature_table(skip_if_exists=True)

    logger.info("Recreating train/validation split")
    X_train, X_val, y_train, y_val = prepare_data(features)

    reference_dir = Path(settings.model_artifact_path).parent
    save_reference_distribution(
        df=X_train,
        output_dir=reference_dir,
        numerical_features=NUMERICAL_FEATURES,
        categorical_features=CATEGORICAL_FEATURES,
    )
    logger.info("Reference regeneration complete")


if __name__ == "__main__":
    main()