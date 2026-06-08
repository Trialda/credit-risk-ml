import logging

from fastapi.testclient import TestClient

logger = logging.getLogger(__name__)

VALID_PAYLOAD = {
    "amt_credit": 500000.0,
    "amt_income_total": 150000.0,
    "days_birth": -12000,
    "days_employed": -2000,
    "cnt_children": 0,
    "amt_annuity": 24700.0,
}


def test_predict_returns_200(client: TestClient) -> None:
    """POST /predict returns HTTP 200 for a valid payload."""
    response = client.post("/predict", json=VALID_PAYLOAD)
    assert response.status_code == 200


def test_predict_returns_risk_score(client: TestClient) -> None:
    """POST /predict returns a risk score in range [0, 1]."""
    response = client.post("/predict", json=VALID_PAYLOAD)
    risk_score = response.json()["risk_score"]
    assert 0.0 <= risk_score <= 1.0


def test_predict_returns_request_id(client: TestClient) -> None:
    """POST /predict returns a non-empty request_id."""
    response = client.post("/predict", json=VALID_PAYLOAD)
    request_id = response.json()["request_id"]
    assert request_id is not None
    assert len(request_id) > 0


def test_predict_logs_inference_to_database(client: TestClient, db_session) -> None:
    """POST /predict writes an inference log entry to the database."""
    from app.models.db import InferenceLog

    response = client.post("/predict", json=VALID_PAYLOAD)
    request_id = response.json()["request_id"]

    log_entry = (
        db_session.query(InferenceLog)
        .filter(InferenceLog.request_id == request_id)
        .first()
    )

    assert log_entry is not None
    assert log_entry.risk_score == 0.42
    assert log_entry.latency_ms > 0


def test_predict_rejects_missing_field(client: TestClient) -> None:
    """POST /predict returns HTTP 422 when a required field is missing."""
    payload = VALID_PAYLOAD.copy()
    del payload["amt_credit"]
    response = client.post("/predict", json=payload)
    assert response.status_code == 422


def test_predict_rejects_negative_income(client: TestClient) -> None:
    """POST /predict returns HTTP 422 when amt_income_total is negative."""
    payload = VALID_PAYLOAD.copy()
    payload["amt_income_total"] = -1000.0
    response = client.post("/predict", json=payload)
    assert response.status_code == 422


def test_predict_rejects_positive_days_birth(client: TestClient) -> None:
    """POST /predict returns HTTP 422 when days_birth is positive."""
    payload = VALID_PAYLOAD.copy()
    payload["days_birth"] = 100
    response = client.post("/predict", json=payload)
    assert response.status_code == 422