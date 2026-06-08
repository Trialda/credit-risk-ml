import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.models.db import init_db
from app.routers import health, predict

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Handle application startup and shutdown."""
    logger.info("Credit Risk API starting up")
    init_db()
    yield
    logger.info("Credit Risk API shutting down")


app = FastAPI(
    title="Credit Risk API",
    description="Credit default risk prediction service.",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health.router)
app.include_router(predict.router)