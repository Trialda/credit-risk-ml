import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from prometheus_client import make_asgi_app

from app.metrics import MODEL_LOADED
from app.models.db import init_db
from app.routers import explain, health, predict
from app.services.explainer import load_explainer
from app.services.predictor import load_model

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Handle application startup and shutdown."""
    logger.info("Credit Risk API starting up")
    init_db()
    try:
        load_model()
        load_explainer()
        MODEL_LOADED.set(1)
    except RuntimeError as e:
        MODEL_LOADED.set(0)
        logger.warning("Model not available at startup: %s", e)
        logger.warning("Predictions will fail until model is loaded")
    yield
    logger.info("Credit Risk API shutting down")


app = FastAPI(
    title="Credit Risk API",
    description="Credit default risk prediction service.",
    version="0.2.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)

metrics_app = make_asgi_app()
app.mount("/metrics", metrics_app)

app.include_router(health.router)
app.include_router(predict.router)
app.include_router(explain.router)