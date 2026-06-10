import logging
from pathlib import Path

import polars as pl
from sqlalchemy import create_engine, text

from ml.config import settings

logger = logging.getLogger(__name__)

RAW_SCHEMA = "raw"

TABLES = {
    "application_train": "application_train.csv",
    "application_test": "application_test.csv",
    "bureau": "bureau.csv",
    "bureau_balance": "bureau_balance.csv",
    "previous_application": "previous_application.csv",
    "installments_payments": "installments_payments.csv",
    "credit_card_balance": "credit_card_balance.csv",
    "pos_cash_balance": "POS_CASH_balance.csv",
}


def ingest_all(data_dir: Path) -> None:
    """Load all Kaggle CSVs (except application_test.csv) into the Postgres raw schema.

    Args:
        data_dir: Path to the directory containing raw CSV files.
    """
    engine = create_engine(settings.database_url)

    _ensure_schema(engine)

    for table_name, filename in TABLES.items():
        if table_name == "application_test":
            logger.info(
                "Skipping application_test, no TARGET column, "
                "not used in training or evaluation"
            )
            continue

        csv_path = data_dir / filename
        if not csv_path.exists():
            logger.warning("File not found, skipping: %s", csv_path)
            continue

        _ingest_table(engine, table_name, csv_path)

    logger.info("Ingestion complete")


def _ensure_schema(engine) -> None:
    """Create the raw schema if it does not exist.

    Args:
        engine: SQLAlchemy engine connected to Postgres.
    """
    with engine.connect() as conn:
        conn.execute(text(f"CREATE SCHEMA IF NOT EXISTS {RAW_SCHEMA}"))
        conn.commit()
    logger.info("Schema '%s' ready", RAW_SCHEMA)


def _ingest_table(engine, table_name: str, csv_path: Path) -> None:
    """Load a single CSV file into a Postgres table.

    Drops and recreates the table on each run — ingestion is idempotent.
    Column names are lowercased for consistency.

    Args:
        engine: SQLAlchemy engine connected to Postgres.
        table_name: Target table name within the raw schema.
        csv_path: Path to the source CSV file.
    """
    logger.info("Ingesting %s → raw.%s", csv_path.name, table_name)

    df = pl.read_csv(
        csv_path,
        infer_schema_length=10000,
        null_values=["", "NA", "NaN", "XNA"],
    )

    df.columns = [col.lower() for col in df.columns]

    row_count = len(df)
    logger.info("Loaded %s rows from %s", f"{row_count:,}", csv_path.name)

    df.write_database(
        table_name=f"{RAW_SCHEMA}.{table_name}",
        connection=str(settings.database_url),
        if_table_exists="replace",
        engine="sqlalchemy",
        engine_options={
            "chunksize": 10000,
            "method": "multi",
        },
    )

    logger.info("Written %s rows to raw.%s", f"{row_count:,}", table_name)