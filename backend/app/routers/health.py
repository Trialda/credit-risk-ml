import logging

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from sqlalchemy import text

from app.models.db import get_db

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/health", tags=["health"])


@router.get("")
def health_check(db: Session = Depends(get_db)) -> dict:
    """Check API and database connectivity.

    Args:
        db: Database session injected by FastAPI dependency injection.

    Returns:
        Dictionary with status of API and database.
    """
    try:
        db.execute(text("SELECT 1"))
        db_status = "ok"
    except Exception as e:
        logger.error("Database health check failed: %s", e)
        db_status = "unavailable"

    return {"api": "ok", "database": db_status}