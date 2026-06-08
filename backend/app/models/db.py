import logging
from datetime import datetime

from sqlalchemy import DateTime, Float, String, Text, create_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, sessionmaker

from app.config import settings

logger = logging.getLogger(__name__)


class Base(DeclarativeBase):
    pass


class InferenceLog(Base):
    """Records every prediction request for monitoring and audit."""

    __tablename__ = "inference_log"
    __table_args__ = {"schema": "app"}

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    request_id: Mapped[str] = mapped_column(String(36), nullable=False, index=True)
    timestamp: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    features_json: Mapped[str] = mapped_column(Text, nullable=False)
    risk_score: Mapped[float] = mapped_column(Float, nullable=False)
    latency_ms: Mapped[float] = mapped_column(Float, nullable=False)


engine = create_engine(settings.database_url, pool_pre_ping=True)
SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False)


def get_db():
    """Yield a database session and ensure it is closed after use.

    Yields:
        Session: SQLAlchemy database session.
    """
    db = SessionLocal()
    try:
        yield db
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


def init_db() -> None:
    """Create all tables if they do not exist.

    In Iteration 3 this is replaced by Alembic migrations.
    """
    try:
        with engine.connect() as conn:
            conn.execute(
                __import__("sqlalchemy").text("CREATE SCHEMA IF NOT EXISTS app")
            )
            conn.commit()
        Base.metadata.create_all(bind=engine)
        logger.info("Database tables initialised")
    except Exception as e:
        logger.error("Failed to initialise database: %s", e)
        raise