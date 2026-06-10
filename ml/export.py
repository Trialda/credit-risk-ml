import logging
import pickle
from pathlib import Path
from typing import Optional

import mlflow
from mlflow.tracking import MlflowClient

from ml.config import settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def export_best_model(
    experiment_name: str,
    metric: str = "auc_roc",
    output_path: Optional[Path] = None,
) -> Path:
    """Export the best model from an MLflow experiment to disk.

    Queries MLflow for the run with the highest value of the given
    metric, loads the model artifact, and writes it to the path
    the backend container reads from at startup.

    Args:
        experiment_name: Name of the MLflow experiment to query.
        metric: Metric to rank runs by. Defaults to auc_roc.
        output_path: Path to write the model artifact. Defaults to
            settings.model_artifact_path.

    Returns:
        Path where the artifact was written.

    Raises:
        ValueError: If no runs are found for the experiment.
    """
    output_path = output_path or Path(settings.model_artifact_path)

    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    client = MlflowClient()

    model = _load_best_model(client, experiment_name, metric)
    _write_artifact(model, output_path)

    return output_path


def _load_best_model(
    client: MlflowClient,
    experiment_name: str,
    metric: str,
) -> object:
    """Load the best model artifact from MLflow.

    Args:
        client: MLflow tracking client.
        experiment_name: Name of the MLflow experiment.
        metric: Metric to rank runs by.

    Returns:
        Fitted sklearn Pipeline loaded from MLflow.

    Raises:
        ValueError: If no runs are found for the experiment.
    """
    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        raise ValueError(
            f"MLflow experiment '{experiment_name}' not found. "
            f"Run train.py first."
        )

    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=[f"metrics.{metric} DESC"],
        max_results=1,
    )

    if not runs:
        raise ValueError(
            f"No runs found for experiment '{experiment_name}'."
        )

    best_run = runs[0]
    logger.info(
        "Best run: %s — %s: %.4f",
        best_run.info.run_id,
        metric,
        best_run.data.metrics.get(metric, 0),
    )

    model_uri = f"runs:/{best_run.info.run_id}/model"
    model = mlflow.sklearn.load_model(model_uri)

    return model


def _write_artifact(model: object, output_path: Path) -> None:
    """Serialize the model to disk at the given path.

    Args:
        model: Fitted sklearn Pipeline to serialize.
        output_path: Path to write the artifact.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "wb") as f:
        pickle.dump(model, f)

    logger.info("Model artifact written to %s", output_path)


if __name__ == "__main__":
    from ml.train import EXPERIMENT_NAME
    export_best_model(experiment_name=EXPERIMENT_NAME)