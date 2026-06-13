import logging
from typing import Optional

import polars as pl
from sqlalchemy import create_engine, text

from ml.config import settings

logger = logging.getLogger(__name__)

FEATURES_SCHEMA = "features"
FEATURES_TABLE = "feature_table"


def build_feature_table(skip_if_exists: bool = False) -> pl.DataFrame:
    """Build the flat feature table by joining all auxiliary tables.

    Reads from the raw schema, aggregates each auxiliary table,
    joins onto application_train, and writes to the features schema.
    
    Args:
        skip_if_exists: If True, load existing feature table from
            Postgres instead of rebuilding. Useful during development
            to skip expensive recomputation.
    
    Returns:
        Feature DataFrame with one row per applicant.
    """
    if skip_if_exists and _feature_table_exists():
        logger.info(
            "Feature table already exists, loading from features.%s",
            FEATURES_TABLE,
        )
        engine = create_engine(settings.get_database_url())
        df = pl.read_database(
            query=f"SELECT * FROM {FEATURES_SCHEMA}.{FEATURES_TABLE}",
            connection=engine.connect(),
        )
        logger.info(
            "Loaded feature table: %s rows, %s columns",
            f"{len(df):,}",
            len(df.columns),
        )
        return df
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


def _feature_table_exists() -> bool:
    """Check if the feature table exists and has rows.

    Returns:
        True if the feature table exists and contains at least one row.
    """
    from sqlalchemy import text
    engine = create_engine(settings.get_database_url())
    with engine.connect() as conn:
        result = conn.execute(text(
            "SELECT EXISTS (SELECT 1 FROM information_schema.tables "
            "WHERE table_schema = :s AND table_name = :t)"
        ), {"s": FEATURES_SCHEMA, "t": FEATURES_TABLE})
        if not result.scalar():
            return False
        result = conn.execute(
            text(f"SELECT COUNT(*) FROM {FEATURES_SCHEMA}.{FEATURES_TABLE}")
        )
        return result.scalar() > 0
    
    
def _load_application() -> pl.DataFrame:
    """Load application_train with derived features from SQL.

    Returns:
        Application DataFrame with basic derived features added.
    """
    query = """
        SELECT
            *,
            amt_credit::DOUBLE PRECISION /
                NULLIF(amt_income_total::DOUBLE PRECISION, 0)
                AS credit_income_ratio,
            amt_annuity::DOUBLE PRECISION /
                NULLIF(amt_income_total::DOUBLE PRECISION, 0)
                AS annuity_income_ratio,
            amt_credit::DOUBLE PRECISION /
                NULLIF(amt_annuity::DOUBLE PRECISION, 0)
                AS credit_term,
            days_birth::DOUBLE PRECISION / -365
                AS age_years,
            days_employed::DOUBLE PRECISION / -365
                AS employment_years,
            days_employed::DOUBLE PRECISION /
                NULLIF(days_birth::DOUBLE PRECISION, 0)
                AS employment_to_age_ratio
        FROM raw.application_train
    """
    engine = create_engine(settings.get_database_url())
    df = pl.read_database(query=query, connection=engine.connect())
    logger.info("Loaded application_train: %s rows", f"{len(df):,}")
    return df


def _aggregate_bureau() -> pl.DataFrame:
    """Aggregate bureau credit history per applicant using SQL.

    Returns:
        DataFrame with one row per applicant, keyed on sk_id_curr.
    """
    query = """
        SELECT
            sk_id_curr,
            COUNT(*) AS bureau_count,
            AVG(days_credit) AS bureau_mean_days_credit,
            AVG(days_credit_enddate) AS bureau_mean_enddate,
            SUM(amt_credit_sum) AS bureau_total_credit,
            SUM(amt_credit_sum_debt) AS bureau_total_debt,
            AVG(amt_credit_sum_overdue) AS bureau_mean_overdue,
            MAX(credit_day_overdue) AS bureau_max_overdue_days,
            SUM(CASE WHEN credit_active = 'Active' THEN 1 ELSE 0 END)
                AS bureau_active_count,
            SUM(CASE WHEN credit_active = 'Closed' THEN 1 ELSE 0 END)
                AS bureau_closed_count
        FROM raw.bureau
        GROUP BY sk_id_curr
    """
    engine = create_engine(settings.get_database_url())
    df = pl.read_database(query=query, connection=engine.connect())
    logger.info("Bureau aggregation: %s rows", f"{len(df):,}")
    return df


def _aggregate_bureau_balance() -> pl.DataFrame:
    """Aggregate bureau balance history using SQL.

    Returns:
        DataFrame with one row per applicant, keyed on sk_id_curr.
    """
    query = """
        SELECT
            b.sk_id_curr,
            COUNT(*) AS bureau_bal_count,
            SUM(CASE WHEN bb.status = 'C' THEN 1 ELSE 0 END)
                AS bureau_bal_closed_count,
            SUM(CASE WHEN bb.status = 'X' THEN 1 ELSE 0 END)
                AS bureau_bal_unknown_count,
            MIN(bb.months_balance) AS bureau_bal_months_min
        FROM raw.bureau_balance bb
        JOIN raw.bureau b ON bb.sk_id_bureau = b.sk_id_bureau
        GROUP BY b.sk_id_curr
    """
    engine = create_engine(settings.get_database_url())
    df = pl.read_database(query=query, connection=engine.connect())
    logger.info("Bureau balance aggregation: %s rows", f"{len(df):,}")
    return df


def _aggregate_previous_applications() -> pl.DataFrame:
    """Aggregate previous applications using SQL.

    Returns:
        DataFrame with one row per applicant, keyed on sk_id_curr.
    """
    query = """
        SELECT
            sk_id_curr,
            COUNT(*) AS prev_app_count,
            SUM(CASE WHEN name_contract_status = 'Approved' THEN 1 ELSE 0 END)
                AS prev_app_approved_count,
            SUM(CASE WHEN name_contract_status = 'Refused' THEN 1 ELSE 0 END)
                AS prev_app_refused_count,
            AVG(amt_credit::DOUBLE PRECISION) AS prev_app_mean_credit,
            AVG(amt_down_payment::DOUBLE PRECISION)
                AS prev_app_mean_down_payment,
            AVG(days_decision::DOUBLE PRECISION)
                AS prev_app_mean_days_decision,
            AVG(cnt_payment::DOUBLE PRECISION) AS prev_app_mean_term
        FROM raw.previous_application
        GROUP BY sk_id_curr
    """
    engine = create_engine(settings.get_database_url())
    df = pl.read_database(query=query, connection=engine.connect())
    logger.info("Previous application aggregation: %s rows", f"{len(df):,}")
    return df


def _aggregate_installments() -> pl.DataFrame:
    """Aggregate installment payment history using SQL.

    Returns:
        DataFrame with one row per applicant, keyed on sk_id_curr.
    """
    query = """
        SELECT
            sk_id_curr,
            COUNT(*) AS installments_count,
            AVG(
                amt_instalment::DOUBLE PRECISION -
                amt_payment::DOUBLE PRECISION
            ) AS installments_mean_payment_diff,
            MAX(
                amt_instalment::DOUBLE PRECISION -
                amt_payment::DOUBLE PRECISION
            ) AS installments_max_payment_diff,
            AVG(
                days_instalment::DOUBLE PRECISION -
                days_entry_payment::DOUBLE PRECISION
            ) AS installments_mean_days_late,
            MAX(
                days_instalment::DOUBLE PRECISION -
                days_entry_payment::DOUBLE PRECISION
            ) AS installments_max_days_late,
            SUM(
                CASE WHEN days_instalment::DOUBLE PRECISION -
                    days_entry_payment::DOUBLE PRECISION > 0
                THEN 1 ELSE 0 END
            ) AS installments_late_count
        FROM raw.installments_payments
        GROUP BY sk_id_curr
    """
    engine = create_engine(settings.get_database_url())
    df = pl.read_database(query=query, connection=engine.connect())
    logger.info("Installments aggregation: %s rows", f"{len(df):,}")
    return df


def _aggregate_credit_card() -> pl.DataFrame:
    """Aggregate credit card balance history using SQL.

    Returns:
        DataFrame with one row per applicant, keyed on sk_id_curr.
    """
    query = """
        SELECT
            sk_id_curr,
            COUNT(*) AS credit_card_count,
            AVG(amt_balance) AS credit_card_mean_balance,
            MAX(amt_balance) AS credit_card_max_balance,
            AVG(
                CASE
                    WHEN amt_credit_limit_actual = 0 THEN NULL
                    ELSE amt_balance / amt_credit_limit_actual
                END
            ) AS credit_card_mean_utilisation,
            MAX(
                CASE
                    WHEN amt_credit_limit_actual = 0 THEN NULL
                    ELSE amt_balance / amt_credit_limit_actual
                END
            ) AS credit_card_max_utilisation,
            SUM(amt_drawings_current) AS credit_card_total_drawings
        FROM raw.credit_card_balance
        GROUP BY sk_id_curr
    """
    engine = create_engine(settings.get_database_url())
    df = pl.read_database(query=query, connection=engine.connect())
    logger.info("Credit card aggregation: %s rows", f"{len(df):,}")
    return df


def _aggregate_pos_cash() -> pl.DataFrame:
    """Aggregate POS and cash loan balance history using SQL.

    Returns:
        DataFrame with one row per applicant, keyed on sk_id_curr.
    """
    query = """
        SELECT
            sk_id_curr,
            COUNT(*) AS pos_cash_count,
            AVG(cnt_instalment::DOUBLE PRECISION)
                AS pos_cash_mean_instalment,
            AVG(sk_dpd::DOUBLE PRECISION) AS pos_cash_mean_dpd,
            MAX(sk_dpd::DOUBLE PRECISION) AS pos_cash_max_dpd,
            AVG(sk_dpd_def::DOUBLE PRECISION) AS pos_cash_mean_dpd_def,
            SUM(CASE WHEN sk_dpd::DOUBLE PRECISION > 0 THEN 1 ELSE 0 END)
                AS pos_cash_late_count
        FROM raw.pos_cash_balance
        GROUP BY sk_id_curr
    """
    engine = create_engine(settings.get_database_url())
    df = pl.read_database(query=query, connection=engine.connect())
    logger.info("POS cash aggregation: %s rows", f"{len(df):,}")
    return df


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
    engine = create_engine(settings.get_database_url())

    with engine.connect() as conn:
        conn.execute(text("CREATE SCHEMA IF NOT EXISTS features"))
        conn.commit()

    df.write_database(
        table_name=f"{FEATURES_SCHEMA}.{FEATURES_TABLE}",
        connection=str(settings.get_database_url()),
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