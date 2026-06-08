import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.routers import health, predict

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Handle application startup and shutdown."""
    logger.info("Credit Risk API starting up")
    yield
    logger.info("Credit Risk API shutting down")


app = FastAPI(
    title="Credit Risk API",
    description="Credit default risk prediction service.",
    version="0.1.0",
    lifespan=lifespan,
)

app.include_router(health.router)
app.include_router(predict.router)