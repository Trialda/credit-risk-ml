import logging
from typing import Tuple

import numpy as np
import pandas as pd
import polars as pl
from sklearn.pipeline import Pipeline, FunctionTransformer

from ml.config import settings

logger = logging.getLogger(__name__)

CATEGORICAL_FEATURES = [
    "name_contract_type",
    "code_gender",
    "name_income_type",
    "name_education_type",
    "name_family_status",
    "name_housing_type",
    "occupation_type",
    "organization_type",
]

NUMERICAL_FEATURES = [
    "amt_credit",
    "amt_income_total",
    "amt_annuity",
    "amt_goods_price",
    "days_birth",
    "days_employed",
    "days_registration",
    "days_id_publish",
    "cnt_children",
    "cnt_fam_members",
    "credit_income_ratio",
    "annuity_income_ratio",
    "credit_term",
    "age_years",
    "employment_years",
    "employment_to_age_ratio",
    "bureau_count",
    "bureau_mean_days_credit",
    "bureau_total_credit",
    "bureau_total_debt",
    "bureau_mean_overdue",
    "bureau_max_overdue_days",
    "bureau_active_count",
    "bureau_closed_count",
    "bureau_bal_count",
    "bureau_bal_closed_count",
    "prev_app_count",
    "prev_app_approved_count",
    "prev_app_refused_count",
    "prev_app_mean_credit",
    "prev_app_mean_term",
    "installments_count",
    "installments_mean_payment_diff",
    "installments_max_payment_diff",
    "installments_mean_days_late",
    "installments_max_days_late",
    "installments_late_count",
    "credit_card_count",
    "credit_card_mean_balance",
    "credit_card_mean_utilisation",
    "credit_card_max_utilisation",
    "pos_cash_count",
    "pos_cash_mean_dpd",
    "pos_cash_max_dpd",
    "pos_cash_late_count",
]

ALL_FEATURES = NUMERICAL_FEATURES + CATEGORICAL_FEATURES
TARGET = "target"


def _cast_categoricals(X: pd.DataFrame) -> pd.DataFrame:
    """Cast categorical columns to category dtype and ensure
    numerical columns are float.

    Args:
        X: Input DataFrame.

    Returns:
        DataFrame with correct dtypes for LightGBM.
    """
    X = X.copy()

    for col in NUMERICAL_FEATURES:
        if col in X.columns:
            X[col] = pd.to_numeric(X[col], errors="coerce")

    for col in CATEGORICAL_FEATURES:
        if col in X.columns:
            X[col] = X[col].astype("category")

    return X


def build_preprocessor() -> Pipeline:
    """Build the sklearn preprocessing pipeline."""
    
    numerical = NUMERICAL_FEATURES.copy()
    categorical = CATEGORICAL_FEATURES.copy()

    def cast_fn(X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        for col in numerical:
            if col in X.columns:
                X[col] = pd.to_numeric(X[col], errors="coerce")
        for col in categorical:
            if col in X.columns:
                X[col] = X[col].astype("category")
        return X

    caster = FunctionTransformer(cast_fn, validate=False)
    caster.set_output(transform="pandas")

    return Pipeline([
        ("cast_categoricals", caster),
    ])


def prepare_data(
    features: pl.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Convert Polars feature DataFrame to train/validation splits.

    Polars → Pandas boundary.

    Args:
        features: Flat feature DataFrame from features.py.

    Returns:
        Tuple of (X_train, X_val, y_train, y_val).
    """
    logger.info("Preparing data for training")

    pandas_df = features.to_pandas()

    available_features = [
        col for col in ALL_FEATURES if col in pandas_df.columns
    ]

    missing = set(ALL_FEATURES) - set(available_features)
    if missing:
        raise ValueError(
            f"Feature contract violated, missing features: {missing}. "
            f"Check the feature engineering pipeline."
        )

    X = pandas_df[available_features]
    y = pandas_df[TARGET]

    from sklearn.model_selection import train_test_split
    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=settings.random_seed,
        stratify=y,
    )

    logger.info(
        "Train: %s rows, Validation: %s rows",
        f"{len(X_train):,}",
        f"{len(X_val):,}",
    )

    return X_train, X_val, y_train, y_val