import json
import logging
import time
import uuid
from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.models.db import InferenceLog, get_db
from app.models.schemas import PredictRequest, PredictResponse

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/predict", tags=["predict"])


def _toy_model(features: dict[str, Any]) -> float:
    """Temporary toy model returning a deterministic score.

    A real model is loaded from artifact storage in Iteration 2.

    Args:
        features: Dictionary of input features.

    Returns:
        Hardcoded risk score for development purposes.
    """
    return 0.42


@router.post("")
def predict(
    request: PredictRequest,
    db: Session = Depends(get_db),
) -> PredictResponse:
    """Generate a credit risk score for a loan application.

    Args:
        request: Validated loan application features.
        db: Database session injected by FastAPI dependency injection.

    Returns:
        Risk score and unique request identifier.
    """
    request_id = str(uuid.uuid4())
    start_time = time.monotonic()

    risk_score = _toy_model(request.model_dump())

    latency_ms = (time.monotonic() - start_time) * 1000

    _log_inference(
        db=db,
        request_id=request_id,
        features=request.model_dump(),
        risk_score=risk_score,
        latency_ms=latency_ms,
    )

    return PredictResponse(request_id=request_id, risk_score=risk_score)


def _log_inference(
    db: Session,
    request_id: str,
    features: dict[str, Any],
    risk_score: float,
    latency_ms: float,
) -> None:
    """Write inference details to the database log.

    Failures are caught and logged without blocking the prediction response.

    Args:
        db: Database session.
        request_id: Unique identifier for this request.
        features: Input features as a dictionary.
        risk_score: Predicted probability of default.
        latency_ms: Time taken to generate the prediction.
    """
    try:
        log_entry = InferenceLog(
            request_id=request_id,
            features_json=json.dumps(features),
            risk_score=risk_score,
            latency_ms=latency_ms,
        )
        db.add(log_entry)
        db.commit()
    except Exception as e:
        logger.error("Failed to log inference %s: %s", request_id, e)
        db.rollback()