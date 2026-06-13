import logging

from fastapi.testclient import TestClient

logger = logging.getLogger(__name__)

VALID_PAYLOAD = {
    "amt_credit": 500000.0,
    "amt_income_total": 150000.0,
    "amt_annuity": 24700.0,
    "amt_goods_price": 450000.0,
    "days_birth": -12000,
    "days_employed": -2000,
    "days_registration": -5000.0,
    "days_id_publish": -3000.0,
    "cnt_children": 0,
    "cnt_fam_members": 2.0,
    "credit_income_ratio": 3.33,
    "annuity_income_ratio": 0.16,
    "credit_term": 20.24,
    "age_years": 32.87,
    "employment_years": 5.47,
    "employment_to_age_ratio": 0.166,
    "name_contract_type": "Cash loans",
    "code_gender": "M",
    "name_income_type": "Working",
    "name_education_type": "Secondary / secondary special",
    "name_family_status": "Married",
    "name_housing_type": "House / apartment",
    "occupation_type": "missing",
    "organization_type": "Business Entity Type 3",
    "bureau_count": 0.0,
    "bureau_mean_days_credit": 0.0,
    "bureau_total_credit": 0.0,
    "bureau_total_debt": 0.0,
    "bureau_mean_overdue": 0.0,
    "bureau_max_overdue_days": 0.0,
    "bureau_active_count": 0.0,
    "bureau_closed_count": 0.0,
    "bureau_bal_count": 0.0,
    "bureau_bal_closed_count": 0.0,
    "prev_app_count": 0.0,
    "prev_app_approved_count": 0.0,
    "prev_app_refused_count": 0.0,
    "prev_app_mean_credit": 0.0,
    "prev_app_mean_term": 0.0,
    "installments_count": 0.0,
    "installments_mean_payment_diff": 0.0,
    "installments_max_payment_diff": 0.0,
    "installments_mean_days_late": 0.0,
    "installments_max_days_late": 0.0,
    "installments_late_count": 0.0,
    "credit_card_count": 0.0,
    "credit_card_mean_balance": 0.0,
    "credit_card_mean_utilisation": 0.0,
    "credit_card_max_utilisation": 0.0,
    "pos_cash_count": 0.0,
    "pos_cash_mean_dpd": 0.0,
    "pos_cash_max_dpd": 0.0,
    "pos_cash_late_count": 0.0,
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
    assert 0.0 <= log_entry.risk_score <= 1.0
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