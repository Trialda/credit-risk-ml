import logging
from fastapi.testclient import TestClient
from tests.test_predict import VALID_PAYLOAD

logger = logging.getLogger(__name__)


def test_explain_returns_200(client: TestClient) -> None:
    """POST /explain returns HTTP 200 for a valid payload."""
    predict_response = client.post("/predict", json=VALID_PAYLOAD)
    request_id = predict_response.json()["request_id"]

    response = client.post("/explain", json={
        "request_id": request_id,
        "features": VALID_PAYLOAD,
    })
    assert response.status_code == 200


def test_explain_returns_risk_score(client: TestClient) -> None:
    """POST /explain returns a risk score in range [0, 1]."""
    predict_response = client.post("/predict", json=VALID_PAYLOAD)
    request_id = predict_response.json()["request_id"]

    response = client.post("/explain", json={
        "request_id": request_id,
        "features": VALID_PAYLOAD,
    })
    assert 0.0 <= response.json()["risk_score"] <= 1.0


def test_explain_returns_shap_features(client: TestClient) -> None:
    """POST /explain returns a non-empty list of SHAP features."""
    predict_response = client.post("/predict", json=VALID_PAYLOAD)
    request_id = predict_response.json()["request_id"]

    response = client.post("/explain", json={
        "request_id": request_id,
        "features": VALID_PAYLOAD,
    })
    assert len(response.json()["shap_features"]) > 0


def test_explain_returns_base_value(client: TestClient) -> None:
    """POST /explain returns a base value."""
    predict_response = client.post("/predict", json=VALID_PAYLOAD)
    request_id = predict_response.json()["request_id"]

    response = client.post("/explain", json={
        "request_id": request_id,
        "features": VALID_PAYLOAD,
    })
    assert "base_value" in response.json()


def test_explain_rejects_missing_request_id(client: TestClient) -> None:
    """POST /explain returns HTTP 422 when request_id is missing."""
    response = client.post("/explain", json={
        "features": VALID_PAYLOAD,
    })
    assert response.status_code == 422