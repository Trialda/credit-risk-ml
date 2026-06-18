import logging

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.metrics import (
    DRIFT_SAMPLE_SIZE,
    FEATURE_PSI,
    SCORE_MEAN,
    SCORE_PSI,
)
from app.models.db import get_db
from app.services.drift import _fetch_production_window, _load_reference, run_drift_computation
import numpy as np

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/drift", tags=["drift"])


@router.post("")
def compute_drift(db: Session = Depends(get_db)) -> dict:
    """Trigger drift computation against the training reference.

    Reads recent inference requests, computes PSI per feature,
    and updates Prometheus gauges. Called on-demand or by a
    scheduled job.

    Args:
        db: Database session.

    Returns:
        Drift computation results with per-feature PSI values.

    Raises:
        HTTPException 503: If drift computation cannot proceed
            (missing reference file or insufficient data).
    """
    results = run_drift_computation(db)

    if results is None:
        raise HTTPException(
            status_code=503,
            detail=(
                "Drift computation unavailable — ensure training reference "
                "exists and sufficient inference data has been collected"
            ),
        )

    for feature, psi in results["feature_psi"].items():
        FEATURE_PSI.labels(feature=feature).set(psi)

    SCORE_PSI.set(results["score_drift"]["score_psi"])
    SCORE_MEAN.set(results["score_drift"]["score_mean"])
    DRIFT_SAMPLE_SIZE.set(results["n_samples"])

    return results

@router.get("/histogram/{feature}")
def get_feature_histogram(
    feature: str,
    db: Session = Depends(get_db),
) -> dict:
    """Return training vs production histograms for a single feature.

    Used by the frontend to render an overlapping histogram comparison,
    visualising exactly what the PSI computation measures.

    Args:
        feature: Feature name to retrieve histograms for.
        db: Database session.

    Returns:
        Bin edges, training proportions, and production proportions
        for the requested feature.

    Raises:
        HTTPException 404: If the feature is not in the reference.
        HTTPException 503: If insufficient production data exists.
    """
    reference = _load_reference()
    if reference is None or feature not in reference.get("numerical", {}):
        raise HTTPException(
            status_code=404,
            detail=f"Feature '{feature}' not found in training reference",
        )

    stats = reference["numerical"][feature]
    bin_edges = stats["bin_edges"]

    production_df = _fetch_production_window(db)
    if production_df is None or feature not in production_df.columns:
        raise HTTPException(
            status_code=503,
            detail="Insufficient production data for this feature",
        )

    prod_values = production_df[feature].astype(float).dropna().values
    prod_counts, _ = np.histogram(prod_values, bins=bin_edges)
    prod_proportions = (prod_counts / len(prod_values)).tolist()

    return {
        "feature": feature,
        "bin_edges": bin_edges,
        "training_proportions": stats["proportions"],
        "production_proportions": prod_proportions,
        "n_samples": len(prod_values),
    }


@router.get("/features")
def list_drift_features() -> dict:
    """List all numerical features available for histogram comparison.

    Returns:
        Sorted list of feature names present in the training reference.
    """
    reference = _load_reference()
    if reference is None:
        raise HTTPException(status_code=503, detail="Training reference unavailable")

    return {"features": sorted(reference.get("numerical", {}).keys())}