import os
from logging.config import fileConfig
from pathlib import Path

from alembic import context
from sqlalchemy import engine_from_config, pool

config = context.config

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

from app.models.db import Base
target_metadata = Base.metadata


def get_database_url() -> str:
    """Read database URL from environment.

    Returns:
        PostgreSQL connection string.

    Raises:
        ValueError: If DATABASE_URL is not set.
    """
    url = os.getenv("DATABASE_URL")
    if not url:
        env_path = Path(__file__).resolve().parent.parent.parent / ".env"
        if env_path.exists():
            from dotenv import load_dotenv
            load_dotenv(env_path)
            url = os.getenv("DATABASE_URL")

    if not url:
        raise ValueError(
            "DATABASE_URL environment variable is not set"
        )
    return url


def run_migrations_offline() -> None:
    """Run migrations in offline mode (no DB connection needed).

    Useful for generating SQL scripts to review before applying.
    """
    url = get_database_url()
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        include_schemas=True,
        version_table_schema="app",
    )

    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """Run migrations in online mode (direct DB connection).

    This is the standard mode used in development and production.
    """
    configuration = config.get_section(config.config_ini_section)
    configuration["sqlalchemy.url"] = get_database_url()

    connectable = engine_from_config(
        configuration,
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            include_schemas=True,
            version_table_schema="app",
        )

        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()