import logging
import warnings
from pathlib import Path

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, early_stopping
from sklearn.pipeline import Pipeline

from ml.config import DATA_DIR, settings
from ml.pipeline.features import build_feature_table
from ml.pipeline.ingest import ingest_all
from ml.pipeline.preprocess import (
    ALL_FEATURES,
    CATEGORICAL_FEATURES,
    build_preprocessor,
    prepare_data,
)
from ml.evaluate import run_evaluation
from ml.schema.validate import validate_feature_dataframe

warnings.filterwarnings("ignore", category=UserWarning)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

EXPERIMENT_NAME = "credit_risk_lgbm"

LGBM_PARAMS = {
    "n_estimators": 1000,
    "learning_rate": 0.05,
    "num_leaves": 31,
    "max_depth": -1,
    "min_child_samples": 20,
    "colsample_bytree": 0.8,
    "subsample": 0.8,
    "subsample_freq": 1,
    "reg_alpha": 0.1,
    "reg_lambda": 0.1,
    "class_weight": "balanced",
    "random_state": settings.random_seed,
    "n_jobs": -1,
    "verbose": -1,
}


def train() -> None:
    """Run the full training pipeline.

    Steps:
        1. Ingest raw CSVs into Postgres raw schema
        2. Build feature table with joins and aggregations
        3. Validate feature schema contract
        4. Split into train/validation sets
        5. Fit preprocessor and LightGBM model
        6. Evaluate and log metrics to MLflow
        7. Save model artifact
    """
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    mlflow.set_experiment(EXPERIMENT_NAME)

    with mlflow.start_run():
        logger.info("Starting training run")

        logger.info("Step 1: Ingesting raw data")
        ingest_all(DATA_DIR)

        logger.info("Step 2: Building feature table")
        features = build_feature_table()

        logger.info("Step 3: Validating feature schema")
        pandas_df = features.to_pandas()
        validate_feature_dataframe(pandas_df)

        logger.info("Step 4: Preparing train/validation split")
        X_train, X_val, y_train, y_val = prepare_data(features)

        logger.info("Step 5: Fitting model")
        model = _fit_model(X_train, y_train, X_val, y_val)

        logger.info("Step 6: Evaluating model")
        y_pred_proba = model.predict_proba(X_val)[:, 1]
        metrics = run_evaluation(y_val, y_pred_proba)
        mlflow.log_metrics({
            k: v for k, v in metrics.items()
            if isinstance(v, float)
        })

        logger.info("Step 7: Logging to MLflow")
        mlflow.log_params(LGBM_PARAMS)
        mlflow.log_param("random_seed", settings.random_seed)
        mlflow.log_param("train_rows", len(X_train))
        mlflow.log_param("val_rows", len(X_val))
        mlflow.log_param("n_features", len(ALL_FEATURES))
        mlflow.log_param("feature_schema_version", "1.0.0")

        logger.info("Step 8: Saving model artifact")
        _save_artifact(model)

        logger.info(
            "Training complete, AUC: %.4f, KS: %.4f",
            metrics["auc_roc"],
            metrics["ks_statistic"],
        )


def _fit_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
) -> Pipeline:
    """Fit the preprocessing pipeline and LightGBM classifier.

    Args:
        X_train: Training features.
        y_train: Training labels.
        X_val: Validation features for early stopping.
        y_val: Validation labels for early stopping.

    Returns:
        Fitted sklearn Pipeline containing preprocessor and model.
    """
    preprocessor = build_preprocessor()
    X_train_transformed = preprocessor.fit_transform(X_train)
    X_val_transformed = preprocessor.transform(X_val)

    lgbm = LGBMClassifier(**LGBM_PARAMS)
    lgbm.fit(
        X_train_transformed,
        y_train,
        categorical_feature=CATEGORICAL_FEATURES,
        eval_set=[(X_val_transformed, y_val)],
        callbacks=[_get_early_stopping_callback()],
    )

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", lgbm),
    ])

    return pipeline


def _get_early_stopping_callback():
    """Build a LightGBM early stopping callback.

    Returns:
        LightGBM early stopping callback stopping after 50 rounds
        without improvement.
    """
    return early_stopping(stopping_rounds=50, verbose=False)


def _save_artifact(model: Pipeline) -> None:
    """Serialize the fitted pipeline to disk and log to MLflow.

    The pipeline contains both the preprocessor and the model, 
    serializing them together guarantees training-serving symmetry.

    Args:
        model: Fitted sklearn Pipeline to serialize.
    """
    import pickle

    artifact_path = Path(settings.model_artifact_path)
    artifact_path.parent.mkdir(parents=True, exist_ok=True)

    with open(artifact_path, "wb") as f:
        pickle.dump(model, f)

    mlflow.sklearn.log_model(model, artifact_path="model")

    logger.info("Model artifact saved to %s", artifact_path)


if __name__ == "__main__":
    train()