import logging
from typing import Optional

import polars as pl
from sqlalchemy import create_engine, text

from ml.config import settings

logger = logging.getLogger(__name__)

FEATURES_SCHEMA = "features"
FEATURES_TABLE = "feature_table"


def build_feature_table() -> pl.DataFrame:
    """Build the flat feature table by joining all auxiliary tables.

    Reads from the raw schema, aggregates each auxiliary table,
    joins onto application_train, and writes to the features schema.

    Returns:
        Feature DataFrame with one row per applicant.
    """
    logger.info("Building feature table")

    application = _load_application()
    bureau_agg = _aggregate_bureau()
    bureau_bal_agg = _aggregate_bureau_balance()
    prev_app_agg = _aggregate_previous_applications()
    installments_agg = _aggregate_installments()
    credit_card_agg = _aggregate_credit_card()
    pos_cash_agg = _aggregate_pos_cash()

    features = _join_all(
        application=application,
        bureau_agg=bureau_agg,
        bureau_bal_agg=bureau_bal_agg,
        prev_app_agg=prev_app_agg,
        installments_agg=installments_agg,
        credit_card_agg=credit_card_agg,
        pos_cash_agg=pos_cash_agg,
    )

    _write_features(features)

    logger.info(
        "Feature table built: %s rows, %s columns",
        f"{len(features):,}",
        len(features.columns),
    )

    return features


def _load_application() -> pl.DataFrame:
    """Load application_train from the raw schema.

    Returns:
        Application DataFrame with basic derived features added.
    """
    engine = create_engine(settings.database_url)

    df = pl.read_database(
        query="SELECT * FROM raw.application_train",
        connection=engine.connect(),
    )

    df = df.with_columns([
        (pl.col("amt_credit") / pl.col("amt_income_total"))
        .alias("credit_income_ratio"),

        (pl.col("amt_annuity") / pl.col("amt_income_total"))
        .alias("annuity_income_ratio"),

        (pl.col("amt_credit") / pl.col("amt_annuity"))
        .alias("credit_term"),

        (pl.col("days_birth") / -365)
        .alias("age_years"),

        (pl.col("days_employed") / -365)
        .alias("employment_years"),

        (pl.col("days_employed") / pl.col("days_birth"))
        .alias("employment_to_age_ratio"),
    ])

    logger.info("Loaded application_train: %s rows", f"{len(df):,}")
    return df


def _aggregate_bureau() -> pl.DataFrame:
    """Aggregate bureau credit history per applicant.

    Returns:
        DataFrame with one row per applicant, keyed on sk_id_curr.
    """
    engine = create_engine(settings.database_url)

    df = pl.read_database(
        query="SELECT * FROM raw.bureau",
        connection=engine.connect(),
    )

    agg = df.group_by("sk_id_curr").agg([
        pl.len().alias("bureau_count"),
        pl.col("days_credit").mean().alias("bureau_mean_days_credit"),
        pl.col("days_credit_enddate").mean().alias("bureau_mean_enddate"),
        pl.col("amt_credit_sum").sum().alias("bureau_total_credit"),
        pl.col("amt_credit_sum_debt").sum().alias("bureau_total_debt"),
        pl.col("amt_credit_sum_overdue").mean().alias("bureau_mean_overdue"),
        pl.col("credit_day_overdue").max().alias("bureau_max_overdue_days"),
        (pl.col("credit_active") == "Active")
        .sum().alias("bureau_active_count"),
        (pl.col("credit_active") == "Closed")
        .sum().alias("bureau_closed_count"),
    ])

    logger.info("Bureau aggregation: %s rows", f"{len(agg):,}")
    return agg


def _aggregate_bureau_balance() -> pl.DataFrame:
    """Aggregate bureau balance history, joining through bureau.

    Returns:
        DataFrame with one row per applicant, keyed on sk_id_curr.
    """
    engine = create_engine(settings.database_url)

    bureau = pl.read_database(
        query="SELECT sk_id_curr, sk_id_bureau FROM raw.bureau",
        connection=engine.connect(),
    )

    balance = pl.read_database(
        query="SELECT * FROM raw.bureau_balance",
        connection=engine.connect(),
    )

    df = balance.join(bureau, on="sk_id_bureau", how="left")

    agg = df.group_by("sk_id_curr").agg([
        pl.len().alias("bureau_bal_count"),
        (pl.col("status") == "C")
        .sum().alias("bureau_bal_closed_count"),
        (pl.col("status") == "X")
        .sum().alias("bureau_bal_unknown_count"),
        pl.col("months_balance").min().alias("bureau_bal_months_min"),
    ])

    logger.info("Bureau balance aggregation: %s rows", f"{len(agg):,}")
    return agg


def _aggregate_previous_applications() -> pl.DataFrame:
    """Aggregate previous Home Credit applications per applicant.

    Returns:
        DataFrame with one row per applicant, keyed on sk_id_curr.
    """
    engine = create_engine(settings.database_url)

    df = pl.read_database(
        query="SELECT * FROM raw.previous_application",
        connection=engine.connect(),
    )

    agg = df.group_by("sk_id_curr").agg([
        pl.len().alias("prev_app_count"),
        (pl.col("name_contract_status") == "Approved")
        .sum().alias("prev_app_approved_count"),
        (pl.col("name_contract_status") == "Refused")
        .sum().alias("prev_app_refused_count"),
        pl.col("amt_credit").mean().alias("prev_app_mean_credit"),
        pl.col("amt_down_payment").mean().alias("prev_app_mean_down_payment"),
        pl.col("days_decision").mean().alias("prev_app_mean_days_decision"),
        pl.col("cnt_payment").mean().alias("prev_app_mean_term"),
    ])

    logger.info("Previous application aggregation: %s rows", f"{len(agg):,}")
    return agg


def _aggregate_installments() -> pl.DataFrame:
    """Aggregate installment payment history per applicant.

    Returns:
        DataFrame with one row per applicant, keyed on sk_id_curr.
    """
    engine = create_engine(settings.database_url)

    df = pl.read_database(
        query="SELECT * FROM raw.installments_payments",
        connection=engine.connect(),
    )

    df = df.with_columns([
        (pl.col("amt_instalment") - pl.col("amt_payment"))
        .alias("payment_diff"),
        (pl.col("days_instalment") - pl.col("days_entry_payment"))
        .alias("days_late"),
    ])

    agg = df.group_by("sk_id_curr").agg([
        pl.len().alias("installments_count"),
        pl.col("payment_diff").mean().alias("installments_mean_payment_diff"),
        pl.col("payment_diff").max().alias("installments_max_payment_diff"),
        pl.col("days_late").mean().alias("installments_mean_days_late"),
        pl.col("days_late").max().alias("installments_max_days_late"),
        (pl.col("days_late") > 0).sum().alias("installments_late_count"),
    ])

    logger.info("Installments aggregation: %s rows", f"{len(agg):,}")
    return agg


def _aggregate_credit_card() -> pl.DataFrame:
    """Aggregate credit card balance history per applicant.

    Returns:
        DataFrame with one row per applicant, keyed on sk_id_curr.
    """
    engine = create_engine(settings.database_url)

    df = pl.read_database(
        query="SELECT * FROM raw.credit_card_balance",
        connection=engine.connect(),
    )

    df = df.with_columns([
        (pl.col("amt_balance") / pl.col("amt_credit_limit_actual")
         .replace(0, None))
        .alias("credit_utilisation"),
    ])

    agg = df.group_by("sk_id_curr").agg([
        pl.len().alias("credit_card_count"),
        pl.col("amt_balance").mean().alias("credit_card_mean_balance"),
        pl.col("amt_balance").max().alias("credit_card_max_balance"),
        pl.col("credit_utilisation").mean()
        .alias("credit_card_mean_utilisation"),
        pl.col("credit_utilisation").max()
        .alias("credit_card_max_utilisation"),
        pl.col("amt_drawings_total").sum()
        .alias("credit_card_total_drawings"),
    ])

    logger.info("Credit card aggregation: %s rows", f"{len(agg):,}")
    return agg


def _aggregate_pos_cash() -> pl.DataFrame:
    """Aggregate POS and cash loan balance history per applicant.

    Returns:
        DataFrame with one row per applicant, keyed on sk_id_curr.
    """
    engine = create_engine(settings.database_url)

    df = pl.read_database(
        query="SELECT * FROM raw.pos_cash_balance",
        connection=engine.connect(),
    )

    agg = df.group_by("sk_id_curr").agg([
        pl.len().alias("pos_cash_count"),
        pl.col("cnt_instalment").mean().alias("pos_cash_mean_instalment"),
        pl.col("sk_dpd").mean().alias("pos_cash_mean_dpd"),
        pl.col("sk_dpd").max().alias("pos_cash_max_dpd"),
        pl.col("sk_dpd_def").mean().alias("pos_cash_mean_dpd_def"),
        (pl.col("sk_dpd") > 0).sum().alias("pos_cash_late_count"),
    ])

    logger.info("POS cash aggregation: %s rows", f"{len(agg):,}")
    return agg


def _join_all(
    application: pl.DataFrame,
    bureau_agg: pl.DataFrame,
    bureau_bal_agg: pl.DataFrame,
    prev_app_agg: pl.DataFrame,
    installments_agg: pl.DataFrame,
    credit_card_agg: pl.DataFrame,
    pos_cash_agg: pl.DataFrame,
) -> pl.DataFrame:
    """Left join all aggregations onto the application table.

    Left joins preserve all applicants even if they have no history
    in a given auxiliary table, missing values are handled in
    preprocess.py via imputation.

    Args:
        application: Base application DataFrame.
        bureau_agg: Aggregated bureau features.
        bureau_bal_agg: Aggregated bureau balance features.
        prev_app_agg: Aggregated previous application features.
        installments_agg: Aggregated installment features.
        credit_card_agg: Aggregated credit card features.
        pos_cash_agg: Aggregated POS cash features.

    Returns:
        Flat DataFrame with one row per applicant.
    """
    df = application

    for agg_df in [
        bureau_agg,
        bureau_bal_agg,
        prev_app_agg,
        installments_agg,
        credit_card_agg,
        pos_cash_agg,
    ]:
        df = df.join(agg_df, on="sk_id_curr", how="left")

    return df


def _write_features(df: pl.DataFrame) -> None:
    """Write the feature table to the features schema in Postgres.

    Args:
        df: Flat feature DataFrame with one row per applicant.
    """
    engine = create_engine(settings.database_url)

    with engine.connect() as conn:
        conn.execute(text("CREATE SCHEMA IF NOT EXISTS features"))
        conn.commit()

    df.write_database(
        table_name=f"{FEATURES_SCHEMA}.{FEATURES_TABLE}",
        connection=str(settings.database_url),
        if_table_exists="replace",
        engine="sqlalchemy",
        engine_options={
            "chunksize": 10000,
            "method": "multi",
        },
    )

    logger.info(
        "Written feature table to features.%s", FEATURES_TABLE
    )