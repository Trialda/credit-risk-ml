import logging

from prometheus_client import Counter, Gauge, Histogram

logger = logging.getLogger(__name__)

PREDICTION_LATENCY = Histogram(
    name="credit_risk_prediction_latency_seconds",
    documentation="Time spent computing a prediction",
    buckets=[0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0],
)

EXPLANATION_LATENCY = Histogram(
    name="credit_risk_explanation_latency_seconds",
    documentation="Time spent computing SHAP explanation",
    buckets=[0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0],
)

PREDICTION_SCORE = Histogram(
    name="credit_risk_prediction_score",
    documentation="Distribution of predicted risk scores",
    buckets=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
)

REQUEST_COUNT = Counter(
    name="credit_risk_requests_total",
    documentation="Total number of prediction requests",
    labelnames=["endpoint", "status"],
)

MODEL_LOADED = Gauge(
    name="credit_risk_model_loaded",
    documentation="Whether the model is currently loaded (1=yes, 0=no)",
)