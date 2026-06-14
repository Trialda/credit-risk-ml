import logging
from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import text
from sqlalchemy.orm import Session

from app.models.db import get_db

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/enrich", tags=["enrich"])


@router.get("/{customer_id}")
def enrich(
    customer_id: int,
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    """Return pre-computed features for a customer from the feature table.

    Used to pre-populate the frontend form. The user can then review
    and edit values before submitting a prediction request.

    Args:
        customer_id: The SK_ID_CURR identifier from the Home Credit dataset.
        db: Database session injected by FastAPI dependency injection.

    Returns:
        Dictionary of feature names to values for the given customer.

    Raises:
        HTTPException 404: If the customer ID is not found in the feature table.
        HTTPException 503: If the feature table cannot be queried.
    """
    try:
        result = db.execute(
            text(
                "SELECT * FROM features.feature_table "
                "WHERE sk_id_curr = :customer_id "
                "LIMIT 1"
            ),
            {"customer_id": customer_id},
        )
        row = result.mappings().first()
    except Exception as e:
        logger.error("Failed to query feature table: %s", e)
        raise HTTPException(
            status_code=503,
            detail="Feature table unavailable",
        )

    if row is None:
        raise HTTPException(
            status_code=404,
            detail=f"Customer {customer_id} not found in feature table",
        )

    features = dict(row)
    features.pop("sk_id_curr", None)
    features.pop("target", None)

    _replace_none_with_zero(features)

    logger.info(
        "Enriched features for customer %d (%d features)",
        customer_id,
        len(features),
    )

    return features


def _replace_none_with_zero(features: dict[str, Any]) -> None:
    """Replace None values with 0 in place.

    Null values in the feature table represent applicants with no
    history in a given auxiliary table. The model handles these as
    missing values natively, but the frontend form needs concrete
    values to display.

    Args:
        features: Feature dictionary to update in place.
    """
    for key, value in features.items():
        if value is None:
            features[key] = 0