import asyncio
import logging
import random
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import httpx
from fastapi import APIRouter, BackgroundTasks, HTTPException
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/simulate", tags=["simulate"])
from app.config import settings

@dataclass
class SimulationState:
    """Tracks the state of a running simulation.

    Attributes:
        running: Whether the simulation is currently active.
        total: Total number of requests to send.
        completed: Number of requests sent so far.
        failed: Number of requests that failed.
        cancelled: Whether the simulation was stopped early.
    """

    running: bool = False
    total: int = 0
    completed: int = 0
    failed: int = 0
    cancelled: bool = False
    current_params: dict = field(default_factory=dict)


_state = SimulationState()


class SimulateRequest(BaseModel):
    """Parameters for a simulation run.

    Attributes:
        n_requests: Total number of synthetic requests to send.
        duration_seconds: Spread requests evenly over this window.
            Set low (e.g. 60) for rate limit testing, high (e.g. 3600)
            for realistic drift accumulation.
        normal_fraction: Proportion of requests drawn from the training
            distribution. The remainder are drifted.
        drift_feature: Which feature to drift. Must be a numerical
            feature present in the feature schema.
        drift_magnitude: How many standard deviations to shift the
            drifted feature. 2.0 = moderate drift, 5.0 = severe.
        drift_speed: 'sudden' applies full drift immediately.
            'gradual' linearly increases drift magnitude over the run.
    """

    n_requests: int = Field(100, ge=1, le=5000)
    duration_seconds: int = Field(60, ge=1, le=86400)
    normal_fraction: float = Field(0.5, ge=0.0, le=1.0)
    drift_feature: str = Field("amt_credit")
    drift_magnitude: float = Field(2.0, ge=0.1, le=10.0)
    drift_speed: str = Field("gradual", pattern="^(sudden|gradual)$")


FEATURE_DISTRIBUTIONS = {
    "amt_credit": {"mean": 599000, "std": 402000, "min": 45000},
    "amt_income_total": {"mean": 168000, "std": 237000, "min": 25650},
    "amt_annuity": {"mean": 27000, "std": 14000, "min": 1615},
    "amt_goods_price": {"mean": 538000, "std": 369000, "min": 40500},
    "days_birth": {"mean": -16000, "std": 4200, "max": -7300},
    "days_employed": {"mean": -2000, "std": 2000},
    "days_registration": {"mean": -4800, "std": 3600},
    "days_id_publish": {"mean": -2900, "std": 1500},
    "cnt_children": {"mean": 0.4, "std": 0.7, "min": 0},
    "cnt_fam_members": {"mean": 2.2, "std": 0.9, "min": 1},
}

CATEGORICAL_DISTRIBUTIONS = {
    "name_contract_type": {
        "Cash loans": 0.90,
        "Revolving loans": 0.10,
    },
    "code_gender": {"M": 0.36, "F": 0.64},
    "name_income_type": {
        "Working": 0.52,
        "Commercial associate": 0.23,
        "Pensioner": 0.18,
        "State servant": 0.07,
    },
    "name_education_type": {
        "Secondary / secondary special": 0.71,
        "Higher education": 0.25,
        "Incomplete higher": 0.03,
        "Lower secondary": 0.01,
    },
    "name_family_status": {
        "Married": 0.64,
        "Single / not married": 0.14,
        "Civil marriage": 0.10,
        "Separated": 0.07,
        "Widow": 0.05,
    },
    "name_housing_type": {
        "House / apartment": 0.88,
        "With parents": 0.05,
        "Municipal apartment": 0.04,
        "Rented apartment": 0.02,
        "Office apartment": 0.01,
    },
    "occupation_type": {"missing": 1.0},
    "organization_type": {"Business Entity Type 3": 1.0},
}


@router.post("")
async def start_simulation(
    params: SimulateRequest,
    background_tasks: BackgroundTasks,
) -> dict:
    """Start a synthetic traffic simulation.

    Sends N predict requests over the specified duration, mixing
    normal and drifted traffic according to the given parameters.

    Args:
        params: Simulation configuration.
        background_tasks: FastAPI background task runner.

    Returns:
        Confirmation with simulation parameters.

    Raises:
        HTTPException 409: If a simulation is already running.
    """
    if _state.running:
        raise HTTPException(
            status_code=409,
            detail="A simulation is already running. "
                   "Send DELETE /simulate to stop it first.",
        )

    _state.running = True
    _state.total = params.n_requests
    _state.completed = 0
    _state.failed = 0
    _state.cancelled = False
    _state.current_params = params.model_dump()

    background_tasks.add_task(_run_simulation, params)

    logger.info(
        "Simulation started: %d requests over %ds, "
        "%.0f%% normal, drift on %s (magnitude=%.1f, speed=%s)",
        params.n_requests,
        params.duration_seconds,
        params.normal_fraction * 100,
        params.drift_feature,
        params.drift_magnitude,
        params.drift_speed,
    )

    return {
        "status": "started",
        "params": params.model_dump(),
    }


@router.delete("")
async def stop_simulation() -> dict:
    """Stop a running simulation.

    Returns:
        Current simulation progress at the time of cancellation.

    Raises:
        HTTPException 404: If no simulation is running.
    """
    if not _state.running:
        raise HTTPException(
            status_code=404,
            detail="No simulation is currently running.",
        )

    _state.cancelled = True
    logger.info(
        "Simulation stop requested at %d/%d requests",
        _state.completed,
        _state.total,
    )

    return {
        "status": "stopping",
        "completed": _state.completed,
        "total": _state.total,
    }


@router.get("/status")
async def simulation_status() -> dict:
    """Poll simulation progress.

    Returns:
        Current simulation state including progress and parameters.
    """
    progress = (
        _state.completed / _state.total
        if _state.total > 0 else 0.0
    )

    return {
        "running": _state.running,
        "completed": _state.completed,
        "total": _state.total,
        "failed": _state.failed,
        "cancelled": _state.cancelled,
        "progress": round(progress, 3),
        "params": _state.current_params,
    }


async def _run_simulation(params: SimulateRequest) -> None:
    """Background task: send synthetic predict requests.

    Generates feature vectors mixing normal and drifted traffic,
    sends them to /predict, and tracks progress in module state.

    Args:
        params: Simulation configuration.
    """
    delay = params.duration_seconds / params.n_requests

    headers = {}
    if settings.api_key:
        headers["X-API-Key"] = settings.api_key

    try:
        async with httpx.AsyncClient(
            base_url="http://nginx:80",
            headers=headers,
            timeout=10.0,
        ) as client:
            tasks = []
            for i in range(params.n_requests):
                if _state.cancelled:
                    break

                is_drifted = random.random() > params.normal_fraction

                if is_drifted and params.drift_speed == "gradual":
                    progress = i / params.n_requests
                    effective_magnitude = params.drift_magnitude * progress
                elif is_drifted:
                    effective_magnitude = params.drift_magnitude
                else:
                    effective_magnitude = 0.0

                features = _generate_features(
                    drift_feature=params.drift_feature if is_drifted else None,
                    drift_magnitude=effective_magnitude,
                )

                tasks.append(_send_request(client, features))

                if len(tasks) >= 20 or i == params.n_requests - 1:
                    results = await asyncio.gather(*tasks, return_exceptions=True)
                    for result in results:
                        if isinstance(result, Exception):
                            _state.failed += 1
                        elif result == 200:
                            _state.completed += 1
                        else:
                            _state.failed += 1
                    tasks = []

                await asyncio.sleep(delay)

    finally:
        _state.running = False
        logger.info(
            "Simulation complete: %d sent, %d failed",
            _state.completed,
            _state.failed,
        )


async def _send_request(
    client: httpx.AsyncClient,
    features: dict,
) -> int:
    """Send a single predict request and return the status code.

    Args:
        client: Shared async HTTP client.
        features: Feature dictionary to send.

    Returns:
        HTTP status code.
    """
    try:
        response = await client.post("/api/predict", json=features)
        return response.status_code
    except Exception as e:
        logger.warning("Request error: %s", e)
        raise
INTEGER_FIELDS = {
    "days_birth", "days_employed", "days_registration",
    "days_id_publish", "cnt_children", "cnt_fam_members"
}

def _generate_features(
    drift_feature: Optional[str] = None,
    drift_magnitude: float = 0.0,
) -> dict:
    """Generate a synthetic feature vector.

    Samples numerical features from their training distributions.
    If drift_feature is specified, shifts that feature's mean by
    drift_magnitude standard deviations.

    Args:
        drift_feature: Feature to drift. None for normal traffic.
        drift_magnitude: Standard deviations to shift the feature.

    Returns:
        Complete feature dictionary ready for POST /predict.
    """
    features = {}

    for feature, dist in FEATURE_DISTRIBUTIONS.items():
        mean = dist["mean"]
        std = dist["std"]

        if feature == drift_feature and drift_magnitude > 0:
            mean = mean + drift_magnitude * std

        value = np.random.normal(mean, std)

        if "min" in dist:
            value = max(value, dist["min"])
        if "max" in dist:
            value = min(value, dist["max"])

        if feature in INTEGER_FIELDS:
            value = int(round(value))
        else:
            value = float(round(value, 2))

        features[feature] = value

    for feature, probs in CATEGORICAL_DISTRIBUTIONS.items():
        categories = list(probs.keys())
        weights = list(probs.values())
        features[feature] = random.choices(categories, weights=weights)[0]

    features["credit_income_ratio"] = (
        features["amt_credit"] / features["amt_income_total"]
        if features["amt_income_total"] > 0 else 0.0
    )
    features["annuity_income_ratio"] = (
        features["amt_annuity"] / features["amt_income_total"]
        if features["amt_income_total"] > 0 else 0.0
    )
    features["credit_term"] = (
        features["amt_credit"] / features["amt_annuity"]
        if features["amt_annuity"] > 0 else 0.0
    )
    features["age_years"] = features["days_birth"] / -365
    features["employment_years"] = features["days_employed"] / -365
    features["employment_to_age_ratio"] = (
        features["days_employed"] / features["days_birth"]
        if features["days_birth"] != 0 else 0.0
    )

    for agg_feature in [
        "bureau_count", "bureau_mean_days_credit", "bureau_total_credit",
        "bureau_total_debt", "bureau_mean_overdue", "bureau_max_overdue_days",
        "bureau_active_count", "bureau_closed_count", "bureau_bal_count",
        "bureau_bal_closed_count", "prev_app_count", "prev_app_approved_count",
        "prev_app_refused_count", "prev_app_mean_credit", "prev_app_mean_term",
        "installments_count", "installments_mean_payment_diff",
        "installments_max_payment_diff", "installments_mean_days_late",
        "installments_max_days_late", "installments_late_count",
        "credit_card_count", "credit_card_mean_balance",
        "credit_card_mean_utilisation", "credit_card_max_utilisation",
        "pos_cash_count", "pos_cash_mean_dpd", "pos_cash_max_dpd",
        "pos_cash_late_count",
    ]:
        features[agg_feature] = 0.0

    return features