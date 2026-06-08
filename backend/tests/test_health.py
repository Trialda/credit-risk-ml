import logging

from fastapi.testclient import TestClient

logger = logging.getLogger(__name__)


def test_health_returns_200(client: TestClient) -> None:
    """GET /health returns HTTP 200."""
    response = client.get("/health")
    assert response.status_code == 200


def test_health_api_status_ok(client: TestClient) -> None:
    """GET /health reports api status as ok."""
    response = client.get("/health")
    assert response.json()["api"] == "ok"


def test_health_database_status_ok(client: TestClient) -> None:
    """GET /health reports database status as ok when DB is reachable."""
    response = client.get("/health")
    assert response.json()["database"] == "ok"