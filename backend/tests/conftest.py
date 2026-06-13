import logging
import os
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

from app.main import app
from app.models.db import Base, get_db
from app.models.schemas import ShapFeature

logger = logging.getLogger(__name__)

TEST_DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://creditrisk:creditrisk@localhost:5432/creditrisk"
)

engine = create_engine(TEST_DATABASE_URL)
TestingSessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False)


@pytest.fixture(scope="session", autouse=True)
def setup_database():
    """Create all tables before tests run and drop them after."""
    with engine.connect() as conn:
        conn.execute(text("CREATE SCHEMA IF NOT EXISTS app"))
        conn.commit()
    Base.metadata.create_all(bind=engine)
    yield
    Base.metadata.drop_all(bind=engine)


@pytest.fixture()
def db_session():
    """Yield a database session that rolls back after each test."""
    connection = engine.connect()
    transaction = connection.begin()
    session = TestingSessionLocal(bind=connection)
    yield session
    session.close()
    transaction.rollback()
    connection.close()


def _mock_predict(features: dict) -> float:
    """Return a fixed score for testing purposes."""
    return 0.35


def _mock_explain(features: dict):
    """Return fixed SHAP values for testing purposes."""
    shap_features = [
        ShapFeature(feature="amt_credit", value=500000.0, shap_value=0.12),
        ShapFeature(feature="days_birth", value=-12000, shap_value=-0.08),
    ]
    return shap_features, -0.5


@pytest.fixture()
def client(db_session):
    """Yield a FastAPI test client with DB and model dependencies overridden."""
    def override_get_db():
        try:
            yield db_session
        finally:
            pass

    app.dependency_overrides[get_db] = override_get_db

    with patch("app.routers.predict.run_predict", side_effect=_mock_predict), \
         patch("app.routers.explain.predict", side_effect=_mock_predict), \
         patch("app.routers.explain.explain", side_effect=_mock_explain):
        with TestClient(app) as test_client:
            yield test_client

    app.dependency_overrides.clear()