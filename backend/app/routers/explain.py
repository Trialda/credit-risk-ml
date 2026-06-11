import json
import logging
import time
import uuid

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.models.db import InferenceLog, get_db
from app.models.schemas import ExplainRequest, ExplainResponse
from app.services.explainer import explain
from app.services.predictor import predict

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/explain", tags=["explain"])


@router.post("")
def explain_prediction(
    request: ExplainRequest,
    db: Session = Depends(get_db),
) -> ExplainResponse:
    """Generate a credit risk score with SHAP feature explanations.

    Slower than /predict due to SHAP computation — use /predict
    when only the score is needed.

    Args:
        request: Request containing request_id and feature payload.
        db: Database session injected by FastAPI dependency injection.

    Returns:
        Risk score, base value, and per-feature SHAP contributions.
    """
    start_time = time.monotonic()

    features = request.features.model_dump()

    try:
        risk_score = predict(features)
        shap_features, base_value = explain(features)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except RuntimeError as e:
        logger.error("Model or explainer error: %s", e)
        raise HTTPException(status_code=503, detail="Model unavailable")

    latency_ms = (time.monotonic() - start_time) * 1000

    _log_inference(
        db=db,
        request_id=request.request_id,
        features=features,
        risk_score=risk_score,
        shap_features=shap_features,
        latency_ms=latency_ms,
    )

    return ExplainResponse(
        request_id=request.request_id,
        risk_score=risk_score,
        base_value=base_value,
        shap_features=shap_features,
    )


def _log_inference(
    db: Session,
    request_id: str,
    features: dict,
    risk_score: float,
    shap_features: list,
    latency_ms: float,
) -> None:
    """Write inference and SHAP values to the database log.

    Failures are caught and logged without blocking the response.

    Args:
        db: Database session.
        request_id: Unique identifier for this request.
        features: Input features as a dictionary.
        risk_score: Predicted probability of default.
        shap_features: List of ShapFeature objects.
        latency_ms: Time taken for the full explain call.
    """
    try:
        log_entry = InferenceLog(
            request_id=request_id,
            features_json=json.dumps(features),
            risk_score=risk_score,
            shap_values_json=json.dumps(
                [
                    {"feature": s.feature, "shap_value": s.shap_value}
                    for s in shap_features
                ]
            ),
            latency_ms=latency_ms,
        )
        db.add(log_entry)
        db.commit()
    except Exception as e:
        logger.error("Failed to log explanation %s: %s", request_id, e)
        db.rollback()