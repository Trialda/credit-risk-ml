import json
import logging
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sqlalchemy import text
from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)

PSI_STABLE = 0.1
PSI_WARNING = 0.2

DRIFT_WINDOW_ROWS = int(os.getenv("DRIFT_WINDOW_ROWS", "1000"))
DRIFT_WINDOW_HOURS = int(os.getenv("DRIFT_WINDOW_HOURS", "24"))

REFERENCE_PATH = (
    Path(os.getenv("MODEL_PATH", "/app/model/model.pkl")).parent
    / "training_reference.json"
)

def compute_psi_from_bins(
    bin_edges: list[float],
    reference_proportions: list[float],
    production_values: np.ndarray,
) -> float:
    """Compute PSI using pre-computed training bin edges and proportions.

    Bins production values into the same edges used for the training
    reference, so the comparison uses identical bucket boundaries —
    no reconstruction or distributional assumptions.

    Args:
        bin_edges: Bin boundaries from the training distribution.
        reference_proportions: Proportion of training data in each bin.
        production_values: Recent production values to compare.

    Returns:
        PSI value. Higher means more drift.
    """
    production_values = production_values[~np.isnan(production_values)]

    if len(production_values) == 0:
        return 0.0

    edges = np.array(bin_edges)
    ref_props = np.array(reference_proportions)

    prod_counts, _ = np.histogram(production_values, bins=edges)
    prod_props = prod_counts / len(production_values)

    ref_props = np.where(ref_props == 0, 1e-4, ref_props)
    prod_props = np.where(prod_props == 0, 1e-4, prod_props)

    return float(np.sum((prod_props - ref_props) * np.log(prod_props / ref_props)))

def compute_psi(
    reference: np.ndarray,
    production: np.ndarray,
    bins: int = 10,
) -> float:
    """Compute Population Stability Index between two distributions.

    Args:
        reference: Values from the training distribution.
        production: Values from recent inference requests.
        bins: Number of bins for discretisation.

    Returns:
        PSI value. Higher means more drift.
    """
    reference = reference[~np.isnan(reference)]
    production = production[~np.isnan(production)]

    if len(reference) == 0 or len(production) == 0:
        return 0.0

    breakpoints = np.percentile(reference, np.linspace(0, 100, bins + 1))
    breakpoints = np.unique(breakpoints)

    if len(breakpoints) < 2:
        return 0.0

    ref_counts, _ = np.histogram(reference, bins=breakpoints)
    prod_counts, _ = np.histogram(production, bins=breakpoints)

    ref_props = ref_counts / len(reference)
    prod_props = prod_counts / len(production)

    ref_props = np.where(ref_props == 0, 1e-4, ref_props)
    prod_props = np.where(prod_props == 0, 1e-4, prod_props)

    return float(np.sum((prod_props - ref_props) * np.log(prod_props / ref_props)))


def _load_reference() -> Optional[dict]:
    """Load training reference distribution from disk.

    Returns:
        Reference distribution dict, or None if not found.
    """
    if not REFERENCE_PATH.exists():
        logger.warning(
            "Training reference not found at %s — "
            "run training pipeline first",
            REFERENCE_PATH,
        )
        return None

    with open(REFERENCE_PATH) as f:
        return json.load(f)


def _fetch_production_window(db: Session) -> Optional[pd.DataFrame]:
    """Fetch recent inference features from the log.

    Uses whichever window captures more rows — DRIFT_WINDOW_ROWS
    or DRIFT_WINDOW_HOURS — to ensure statistical significance.

    Args:
        db: Database session.

    Returns:
        DataFrame of recent feature vectors, or None if insufficient data.
    """
    cutoff = datetime.utcnow() - timedelta(hours=DRIFT_WINDOW_HOURS)

    result = db.execute(text("""
        SELECT features_json, risk_score, timestamp
        FROM app.inference_log
        WHERE timestamp >= :cutoff
        ORDER BY timestamp DESC
    """), {"cutoff": cutoff})

    rows = result.fetchall()

    if len(rows) < DRIFT_WINDOW_ROWS:
        result = db.execute(text("""
            SELECT features_json, risk_score, timestamp
            FROM app.inference_log
            ORDER BY timestamp DESC
            LIMIT :limit
        """), {"limit": DRIFT_WINDOW_ROWS})
        rows = result.fetchall()

    if len(rows) < 100:
        logger.info(
            "Insufficient data for drift computation: %d rows "
            "(minimum 100, recommended %d)",
            len(rows),
            DRIFT_WINDOW_ROWS,
        )
        return None

    features_list = []
    scores = []

    for row in rows:
        try:
            features = json.loads(row.features_json)
            features["_risk_score"] = row.risk_score
            features_list.append(features)
            scores.append(row.risk_score)
        except (json.JSONDecodeError, AttributeError) as e:
            logger.warning("Failed to parse inference log row: %s", e)
            continue

    if not features_list:
        return None

    df = pd.DataFrame(features_list)
    logger.info(
        "Fetched %d rows for drift computation (window: %dh or %d rows)",
        len(df),
        DRIFT_WINDOW_HOURS,
        DRIFT_WINDOW_ROWS,
    )
    return df


def run_drift_computation(db: Session) -> Optional[dict]:
    """Compute PSI drift metrics against the training reference.

    Reads recent inference requests, computes per-feature PSI against
    the stored training distribution, and returns results for
    Prometheus metric updates.

    Args:
        db: Database session.

    Returns:
        Dictionary with per-feature PSI values and prediction drift
        metrics, or None if computation cannot proceed.
    """
    reference = _load_reference()
    if reference is None:
        return None

    production_df = _fetch_production_window(db)
    if production_df is None:
        return None

    numerical_stats = reference.get("numerical", {})
    available_features = [
        f for f in numerical_stats
        if f in production_df.columns
    ]

    feature_psi = {}
    for feature in available_features:
        stats = numerical_stats[feature]

        if "bin_edges" not in stats:
            logger.warning(
                "Feature %s missing bin_edges — retrain to update "
                "reference format", feature
            )
            continue

        prod_values = production_df[feature].astype(float).values
        psi = compute_psi_from_bins(
            stats["bin_edges"],
            stats["proportions"],
            prod_values,
        )
        feature_psi[feature] = psi

    scores = production_df["_risk_score"].astype(float).values
    score_psi = 0.0
    score_drift = {
        "score_psi": score_psi,
        "score_mean": float(np.mean(scores)),
    }

    results = {
        "feature_psi": feature_psi,
        "score_drift": score_drift,
        "n_samples": len(production_df),
    }

    _log_drift_summary(feature_psi)
    return results


def _log_drift_summary(feature_psi: dict[str, float]) -> None:
    """Log a summary of drift levels across features.

    Args:
        feature_psi: Per-feature PSI values.
    """
    drifted = [f for f, v in feature_psi.items() if v >= PSI_WARNING]
    warning = [
        f for f, v in feature_psi.items()
        if PSI_STABLE <= v < PSI_WARNING
    ]
    stable = [f for f, v in feature_psi.items() if v < PSI_STABLE]

    logger.info(
        "Drift summary — stable: %d, warning: %d, drift: %d",
        len(stable), len(warning), len(drifted),
    )

    if drifted:
        logger.warning("Features with significant drift: %s", drifted)
    if warning:
        logger.info("Features with moderate drift: %s", warning)