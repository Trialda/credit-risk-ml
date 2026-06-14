import logging
from typing import Optional

import pandas as pd
import pandera as pa
from pandera import Column, DataFrameSchema, Check

logger = logging.getLogger(__name__)

feature_schema = DataFrameSchema(
    columns={
        "amt_credit": Column(float, Check.gt(0), nullable=True),
        "amt_income_total": Column(float, Check.gt(0), nullable=True),
        "amt_annuity": Column(float, Check.gt(0), nullable=True),
        "amt_goods_price": Column(float, Check.ge(0), nullable=True),
        "days_birth": Column(float, Check.lt(0), nullable=True),
        "days_employed": Column(float, nullable=True),
        "days_registration": Column(float, nullable=True),
        "days_id_publish": Column(float, nullable=True),
        "cnt_children": Column(float, Check.ge(0), nullable=True),
        "cnt_fam_members": Column(float, Check.ge(0), nullable=True),
        "credit_income_ratio": Column(float, nullable=True),
        "annuity_income_ratio": Column(float, nullable=True),
        "credit_term": Column(float, nullable=True),
        "age_years": Column(float, nullable=True),
        "employment_years": Column(float, nullable=True),
        "employment_to_age_ratio": Column(float, nullable=True),
        "bureau_count": Column(float, Check.ge(0), nullable=True),
        "bureau_total_credit": Column(float, nullable=True),
        "bureau_total_debt": Column(float, nullable=True),
        "bureau_mean_overdue": Column(float, nullable=True),
        "bureau_max_overdue_days": Column(float, Check.ge(0), nullable=True),
        "bureau_active_count": Column(float, Check.ge(0), nullable=True),
        "bureau_closed_count": Column(float, Check.ge(0), nullable=True),
        "prev_app_count": Column(float, Check.ge(0), nullable=True),
        "prev_app_approved_count": Column(float, Check.ge(0), nullable=True),
        "prev_app_refused_count": Column(float, Check.ge(0), nullable=True),
        "installments_count": Column(float, Check.ge(0), nullable=True),
        "installments_late_count": Column(float, Check.ge(0), nullable=True),
        "credit_card_count": Column(float, Check.ge(0), nullable=True),
        "pos_cash_count": Column(float, Check.ge(0), nullable=True),
        "pos_cash_max_dpd": Column(float, Check.ge(0), nullable=True),
        "pos_cash_late_count": Column(float, Check.ge(0), nullable=True),
    },
    strict=False,
)


def validate_features(features: dict) -> Optional[str]:
    """Validate a feature dictionary against the Pandera schema.

    Called after Pydantic validates the API request shape and before
    the features are passed to the model. Catches domain violations
    that Pydantic cannot, values that are the right type but
    statistically impossible in the credit risk domain.

    Args:
        features: Feature dictionary from the API request.

    Returns:
        None if validation passes. Error message string if it fails.
    """
    try:
        df = pd.DataFrame([features])
        feature_schema.validate(df)
        return None
    except pa.errors.SchemaError as e:
        logger.warning("Pandera validation failed: %s", e)
        return str(e)