import logging

from fastapi.testclient import TestClient

logger = logging.getLogger(__name__)

KNOWN_CUSTOMER_ID = 100002
UNKNOWN_CUSTOMER_ID = 999999999


def test_enrich_returns_200_for_known_customer(client: TestClient) -> None:
    """GET /enrich/{id} returns HTTP 200 for a customer in the feature table."""
    response = client.get(f"/enrich/{KNOWN_CUSTOMER_ID}")
    assert response.status_code == 200


def test_enrich_returns_feature_dict(client: TestClient) -> None:
    """GET /enrich/{id} returns a non-empty feature dictionary."""
    response = client.get(f"/enrich/{KNOWN_CUSTOMER_ID}")
    data = response.json()
    assert isinstance(data, dict)
    assert len(data) > 0


def test_enrich_excludes_target_column(client: TestClient) -> None:
    """GET /enrich/{id} does not return the target label."""
    response = client.get(f"/enrich/{KNOWN_CUSTOMER_ID}")
    data = response.json()
    assert "target" not in data


def test_enrich_excludes_customer_id(client: TestClient) -> None:
    """GET /enrich/{id} does not return sk_id_curr in the response."""
    response = client.get(f"/enrich/{KNOWN_CUSTOMER_ID}")
    data = response.json()
    assert "sk_id_curr" not in data


def test_enrich_returns_404_for_unknown_customer(client: TestClient) -> None:
    """GET /enrich/{id} returns HTTP 404 for a non-existent customer."""
    response = client.get(f"/enrich/{UNKNOWN_CUSTOMER_ID}")
    assert response.status_code == 404


def test_enrich_returns_numeric_values(client: TestClient) -> None:
    """GET /enrich/{id} returns numeric values for key features."""
    response = client.get(f"/enrich/{KNOWN_CUSTOMER_ID}")
    data = response.json()
    assert isinstance(data.get("amt_credit"), (int, float))
    assert isinstance(data.get("amt_income_total"), (int, float))