import logging
from fastapi.testclient import TestClient
from tests.test_predict import VALID_PAYLOAD

logger = logging.getLogger(__name__)


def test_explain_returns_200(client: TestClient) -> None:
    """POST /explain returns HTTP 200 for a valid payload."""
    response = client.post("/explain", json={
        "features": VALID_PAYLOAD,
    })
    assert response.status_code == 200


def test_explain_returns_risk_score(client: TestClient) -> None:
    """POST /explain returns a risk score in range [0, 1]."""
    response = client.post("/explain", json={
        "features": VALID_PAYLOAD,
    })
    assert 0.0 <= response.json()["risk_score"] <= 1.0


def test_explain_returns_shap_features(client: TestClient) -> None:
    """POST /explain returns a non-empty list of SHAP features."""
    response = client.post("/explain", json={
    "features": VALID_PAYLOAD,
    })
    assert len(response.json()["shap_features"]) > 0


def test_explain_returns_base_value(client: TestClient) -> None:
    """POST /explain returns a base value."""
    response = client.post("/explain", json={
        "features": VALID_PAYLOAD,
    })
    assert "base_value" in response.json()


def test_explain_rejects_missing_request_id(client: TestClient) -> None:
    """POST /explain returns HTTP 422 when request_id is missing."""
    response = client.post("/explain", json={
        "features": VALID_PAYLOAD,
    })
    assert response.status_code == 422