from typing import Any
from pydantic import BaseModel, Field


# class PredictRequest(BaseModel):
#     """Input features for a credit risk prediction request.

#     All fields represent applicant and loan characteristics.
#     Feature set is intentionally minimal in Iteration 1, expanded
#     in Iteration 2 to match the full engineered feature schema.
#     """

#     amt_credit: float = Field(..., gt=0, description="Loan credit amount")
#     amt_income_total: float = Field(..., gt=0, description="Applicant annual income")
#     days_birth: int = Field(..., lt=0, description="Age in days (negative integer)")
#     days_employed: int = Field(..., description="Employment duration in days (negative means currently employed)")
#     cnt_children: int = Field(..., ge=0, description="Number of children")
#     amt_annuity: float = Field(..., gt=0, description="Loan annuity amount")

#     model_config = {"json_schema_extra": {"example": {
#         "amt_credit": 500000.0,
#         "amt_income_total": 150000.0,
#         "days_birth": -12000,
#         "days_employed": -2000,
#         "cnt_children": 0,
#         "amt_annuity": 24700.0,
#     }}}
class PredictRequest(BaseModel):
    """Input features for a credit risk prediction request.

    Application-level features are entered manually. Aggregated
    features default to zero — representing an applicant with no
    prior credit history. Feature enrichment from Postgres
    aggregations is added in Iteration 3.
    """

    # Core application features
    amt_credit: float = Field(..., gt=0)
    amt_income_total: float = Field(..., gt=0)
    amt_annuity: float = Field(..., gt=0)
    amt_goods_price: float = Field(0.0, ge=0)
    days_birth: int = Field(..., lt=0)
    days_employed: int = Field(-1000)
    days_registration: float = Field(-1000.0)
    days_id_publish: float = Field(-1000.0)
    cnt_children: int = Field(0, ge=0)
    cnt_fam_members: float = Field(1.0, ge=0)

    # Derived application features — computed from above
    credit_income_ratio: float = Field(0.0)
    annuity_income_ratio: float = Field(0.0)
    credit_term: float = Field(0.0)
    age_years: float = Field(0.0)
    employment_years: float = Field(0.0)
    employment_to_age_ratio: float = Field(0.0)

    # Categorical features
    name_contract_type: str = Field("Cash loans")
    code_gender: str = Field("M")
    name_income_type: str = Field("Working")
    name_education_type: str = Field("Secondary / secondary special")
    name_family_status: str = Field("Married")
    name_housing_type: str = Field("House / apartment")
    occupation_type: str = Field("missing")
    organization_type: str = Field("Business Entity Type 3")

    # Bureau aggregates — default to 0 (no prior credit history)
    bureau_count: float = Field(0.0)
    bureau_mean_days_credit: float = Field(0.0)
    bureau_total_credit: float = Field(0.0)
    bureau_total_debt: float = Field(0.0)
    bureau_mean_overdue: float = Field(0.0)
    bureau_max_overdue_days: float = Field(0.0)
    bureau_active_count: float = Field(0.0)
    bureau_closed_count: float = Field(0.0)
    bureau_bal_count: float = Field(0.0)
    bureau_bal_closed_count: float = Field(0.0)

    # Previous application aggregates
    prev_app_count: float = Field(0.0)
    prev_app_approved_count: float = Field(0.0)
    prev_app_refused_count: float = Field(0.0)
    prev_app_mean_credit: float = Field(0.0)
    prev_app_mean_term: float = Field(0.0)

    # Installment aggregates
    installments_count: float = Field(0.0)
    installments_mean_payment_diff: float = Field(0.0)
    installments_max_payment_diff: float = Field(0.0)
    installments_mean_days_late: float = Field(0.0)
    installments_max_days_late: float = Field(0.0)
    installments_late_count: float = Field(0.0)

    # Credit card aggregates
    credit_card_count: float = Field(0.0)
    credit_card_mean_balance: float = Field(0.0)
    credit_card_mean_utilisation: float = Field(0.0)
    credit_card_max_utilisation: float = Field(0.0)

    # POS cash aggregates
    pos_cash_count: float = Field(0.0)
    pos_cash_mean_dpd: float = Field(0.0)
    pos_cash_max_dpd: float = Field(0.0)
    pos_cash_late_count: float = Field(0.0)

    model_config = {"json_schema_extra": {"example": {
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
    }}}


class PredictResponse(BaseModel):
    """Response returned after a prediction request.

    Attributes:
        request_id: UUID identifying this prediction for log correlation.
        risk_score: Predicted probability of default in range [0, 1].
    """

    request_id: str
    risk_score: float = Field(..., ge=0.0, le=1.0)


class ExplainRequest(BaseModel):
    """Request body for the explain endpoint.

    Attributes:
        request_id: UUID of a prior prediction to explain.
        features: The same feature payload sent to /predict.
    """

    request_id: str
    features: PredictRequest


class ShapFeature(BaseModel):
    """A single feature's SHAP contribution.

    Attributes:
        feature: Feature name.
        value: Raw feature value.
        shap_value: SHAP contribution to the prediction.
    """

    feature: str
    value: Any
    shap_value: float


class ExplainResponse(BaseModel):
    """Response returned after an explanation request.

    Attributes:
        request_id: UUID correlating this explanation to a prediction.
        risk_score: Predicted probability of default.
        base_value: Model base value (mean prediction over training set).
        shap_features: Per-feature SHAP contributions sorted by
            absolute impact, descending.
    """

    request_id: str
    risk_score: float = Field(..., ge=0.0, le=1.0)
    base_value: float
    shap_features: list[ShapFeature]