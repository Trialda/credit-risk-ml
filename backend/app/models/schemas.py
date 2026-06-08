from pydantic import BaseModel, Field


class PredictRequest(BaseModel):
    """Input features for a credit risk prediction request.

    All fields represent applicant and loan characteristics.
    Feature set is intentionally minimal in Iteration 1, expanded
    in Iteration 2 to match the full engineered feature schema.
    """

    amt_credit: float = Field(..., gt=0, description="Loan credit amount")
    amt_income_total: float = Field(..., gt=0, description="Applicant annual income")
    days_birth: int = Field(..., lt=0, description="Age in days (negative integer)")
    days_employed: int = Field(..., description="Employment duration in days (negative means currently employed)")
    cnt_children: int = Field(..., ge=0, description="Number of children")
    amt_annuity: float = Field(..., gt=0, description="Loan annuity amount")

    model_config = {"json_schema_extra": {"example": {
        "amt_credit": 500000.0,
        "amt_income_total": 150000.0,
        "days_birth": -12000,
        "days_employed": -2000,
        "cnt_children": 0,
        "amt_annuity": 24700.0,
    }}}


class PredictResponse(BaseModel):
    """Response returned after a prediction request.

    Attributes:
        request_id: UUID identifying this prediction for log correlation.
        risk_score: Predicted probability of default in range [0, 1].
    """

    request_id: str
    risk_score: float = Field(..., ge=0.0, le=1.0)