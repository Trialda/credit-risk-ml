import pytest
from unittest.mock import patch
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
import os

from app.main import app
from app.models.db import Base, get_db

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


@pytest.fixture()
def client(db_session):
    """Yield a FastAPI test client with DB and model dependencies overridden."""
    def override_get_db():
        try:
            yield db_session
        finally:
            pass

    def mock_predict(features: dict) -> float:
        """Return a fixed score for testing purposes."""
        return 0.35

    app.dependency_overrides[get_db] = override_get_db

    with patch("app.routers.predict.run_predict", side_effect=mock_predict), \
         patch("app.routers.explain.predict", side_effect=mock_predict):
        with TestClient(app) as test_client:
            yield test_client

    app.dependency_overrides.clear()