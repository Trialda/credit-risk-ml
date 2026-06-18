import asyncio
import logging
import random
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import httpx
from fastapi import APIRouter, BackgroundTasks, HTTPException
from pydantic import BaseModel, Field

import os
import json
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/simulate", tags=["simulate"])
from app.config import settings

REFERENCE_PATH = (
    Path(os.getenv("MODEL_PATH", "/app/model/model.pkl")).parent
    / "training_reference.json"
)

SIMULATION_POOL_PATH = Path("/app/ml/data/simulation_pool.csv")
_simulation_pool_cache: Optional[pd.DataFrame] = None

_reference_cache: Optional[dict] = None

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
    data_source: str = Field("synthetic", pattern="^(synthetic|real)$")


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

                if params.data_source == "real":
                    features = _sample_real_row(
                        drift_feature=params.drift_feature if is_drifted else None,
                        drift_magnitude=effective_magnitude,
                    )
                    if features is None:
                        # Pool unavailable — fall back to synthetic
                        features = _generate_features(
                            drift_feature=params.drift_feature if is_drifted else None,
                            drift_magnitude=effective_magnitude,
                        )
                else:
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
            "Simulation complete: %d sent, %d failed (data_source=%s)",
            _state.completed, _state.failed, params.data_source,
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
    reference = _load_reference_for_simulation()

    application_features = [
        "amt_credit", "amt_income_total", "amt_annuity", "amt_goods_price",
        "days_birth", "days_employed", "days_registration", "days_id_publish",
        "cnt_children", "cnt_fam_members",
    ]

    for feature in application_features:
        if reference and feature in reference.get("numerical", {}):
            stats = reference["numerical"][feature]

            if feature == drift_feature and drift_magnitude > 0:
                # Shift the distribution by sampling then adding a
                # magnitude * std offset, preserving empirical shape
                # while still injecting a controllable directional drift
                value = _sample_from_reference(stats)
                value = value + drift_magnitude * stats["std"]
                value = max(min(value, stats["max"] * 2), stats["min"])
            else:
                value = _sample_from_reference(stats)
        else:
            # Fallback to the old Gaussian approach if reference missing
            dist = FEATURE_DISTRIBUTIONS.get(feature, {"mean": 0, "std": 1})
            value = np.random.normal(dist["mean"], dist["std"])

        if feature in INTEGER_FIELDS:
            value = int(round(value))
        else:
            value = float(round(value, 2))

        features[feature] = value

    for feature, probs in CATEGORICAL_DISTRIBUTIONS.items():
        categories = list(probs.keys())
        weights = list(probs.values())
        features[feature] = random.choices(categories, weights=weights)[0]

    # Derived features unchanged — still computed from the sampled values above
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

    # Aggregated features — unchanged, already using histogram sampling
    aggregated_features = [
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
    ]

    for agg_feature in aggregated_features:
        if reference and agg_feature in reference.get("numerical", {}):
            stats = reference["numerical"][agg_feature]
            features[agg_feature] = round(_sample_from_reference(stats), 2)
        else:
            features[agg_feature] = 0.0

    return features

def _load_reference_for_simulation() -> Optional[dict]:
    """Load training reference distribution, caching after first read.

    Returns:
        Reference distribution dict, or None if not found.
    """
    global _reference_cache
    if _reference_cache is not None:
        return _reference_cache

    if not REFERENCE_PATH.exists():
        logger.warning(
            "Training reference not found — aggregated features "
            "will default to 0 in simulation"
        )
        return None

    with open(REFERENCE_PATH) as f:
        _reference_cache = json.load(f)
    return _reference_cache


def _sample_from_reference(stats: dict) -> float:
    """Sample a plausible value from the stored training histogram.

    Picks a bin according to its real training proportion, then samples
    uniformly within that bin's edges. This reproduces the actual shape
    of the training distribution, including zero-inflation and skew,
    rather than assuming a Gaussian, which badly misrepresents features
    like installment payment differences or account balances.

    Args:
        stats: Feature statistics dict with bin_edges and proportions.

    Returns:
        A plausible sampled value matching the training distribution shape.
    """
    if "bin_edges" not in stats or "proportions" not in stats:
        # Fallback for any feature missing histogram data
        value = np.random.normal(stats["mean"], stats["std"])
        return float(np.clip(value, stats["min"], stats["max"]))

    bin_edges = stats["bin_edges"]
    proportions = stats["proportions"]

    bin_index = np.random.choice(len(proportions), p=proportions)
    low = bin_edges[bin_index]
    high = bin_edges[bin_index + 1]

    return float(np.random.uniform(low, high))

def _load_simulation_pool() -> Optional[pd.DataFrame]:
    """Load the real held-out applicant pool, caching after first read."""
    global _simulation_pool_cache
    if _simulation_pool_cache is not None:
        return _simulation_pool_cache
    
    if not SIMULATION_POOL_PATH.exists():
        logger.warning(
            "Simulation pool not found at %s — 'real' data source "
            "unavailable, falling back to synthetic", SIMULATION_POOL_PATH
        )
        return None

    _simulation_pool_cache = pd.read_csv(SIMULATION_POOL_PATH)
    #print(f"=== SIMULATION POOL LOADED: {len(_simulation_pool_cache)} rows ===", flush=True)
    logger.info(
        "Loaded simulation pool: %d real applicant rows",
        len(_simulation_pool_cache),
    )
    return _simulation_pool_cache

DERIVED_FROM = {
    "amt_credit": ["credit_income_ratio", "credit_term"],
    "amt_annuity": ["annuity_income_ratio", "credit_term"],
    "amt_income_total": ["credit_income_ratio", "annuity_income_ratio"],
    "days_birth": ["age_years", "employment_to_age_ratio"],
    "days_employed": ["employment_years", "employment_to_age_ratio"],
}

def _recompute_derived(row: dict, changed_feature: str) -> dict:
    """Recompute ratio features that depend on a mutated feature.

    Args:
        row: Feature dictionary with one feature already mutated.
        changed_feature: The feature that was overwritten.

    Returns:
        Row with dependent derived features recalculated.
    """
    if changed_feature not in DERIVED_FROM:
        return row

    if row.get("amt_income_total", 0) > 0:
        row["credit_income_ratio"] = row["amt_credit"] / row["amt_income_total"]
        row["annuity_income_ratio"] = row["amt_annuity"] / row["amt_income_total"]
    if row.get("amt_annuity", 0) > 0:
        row["credit_term"] = row["amt_credit"] / row["amt_annuity"]
    row["age_years"] = row["days_birth"] / -365
    row["employment_years"] = row["days_employed"] / -365
    if row.get("days_birth", 0) != 0:
        row["employment_to_age_ratio"] = row["days_employed"] / row["days_birth"]

    return row

def _sample_real_row(
    drift_feature: Optional[str] = None,
    drift_magnitude: float = 0.0,
) -> Optional[dict]:
    """Sample one real applicant row from the held-out pool.

    Args:
        drift_feature: Feature to overwrite with a drifted value.
        drift_magnitude: Standard deviations to shift, if drifting.

    Returns:
        Feature dictionary ready for POST /predict, or None if the
        pool is unavailable.
    """
    pool = _load_simulation_pool()
    if pool is None:
        return None

    row = pool.sample(n=1).iloc[0].to_dict()
    row.pop("sk_id_curr", None)

    if drift_feature and drift_magnitude > 0:
        reference = _load_reference_for_simulation()
        if reference and drift_feature in reference.get("numerical", {}):
            std = reference["numerical"][drift_feature]["std"]
            row[drift_feature] = row[drift_feature] + drift_magnitude * std
            row = _recompute_derived(row, drift_feature)

    for field in INTEGER_FIELDS:
        if field in row:
            row[field] = int(round(row[field]))

    return row