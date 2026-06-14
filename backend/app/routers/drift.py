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
from app.services.drift import run_drift_computation

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