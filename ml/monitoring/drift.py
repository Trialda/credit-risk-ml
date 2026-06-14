import logging
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

PSI_STABLE = 0.1
PSI_WARNING = 0.2


def compute_psi(
    reference: np.ndarray,
    production: np.ndarray,
    bins: int = 10,
) -> float:
    """Compute Population Stability Index between two distributions.

    PSI measures how much a distribution has shifted:
        PSI < 0.1  — stable, no action needed
        PSI 0.1-0.2 — moderate shift, investigate
        PSI > 0.2  — significant drift, consider retraining

    Args:
        reference: Values from the training distribution.
        production: Values from recent inference requests.
        bins: Number of bins for discretisation. Defaults to 10.

    Returns:
        PSI value. Higher means more drift.
    """
    reference = reference[~np.isnan(reference)]
    production = production[~np.isnan(production)]

    if len(reference) == 0 or len(production) == 0:
        logger.warning("Empty array passed to compute_psi — returning 0")
        return 0.0

    breakpoints = np.percentile(reference, np.linspace(0, 100, bins + 1))
    breakpoints = np.unique(breakpoints)

    if len(breakpoints) < 2:
        logger.warning(
            "Insufficient unique breakpoints for PSI computation — "
            "feature may be constant"
        )
        return 0.0

    ref_counts, _ = np.histogram(reference, bins=breakpoints)
    prod_counts, _ = np.histogram(production, bins=breakpoints)

    ref_proportions = ref_counts / len(reference)
    prod_proportions = prod_counts / len(production)

    ref_proportions = np.where(ref_proportions == 0, 1e-4, ref_proportions)
    prod_proportions = np.where(prod_proportions == 0, 1e-4, prod_proportions)

    psi = np.sum(
        (prod_proportions - ref_proportions)
        * np.log(prod_proportions / ref_proportions)
    )

    return float(psi)


def compute_psi_for_features(
    reference_stats: dict,
    production_df: pd.DataFrame,
    features: Optional[list[str]] = None,
) -> dict[str, float]:
    """Compute PSI for multiple numerical features.

    Args:
        reference_stats: Training distribution statistics loaded from
            the reference JSON. Must contain a 'numerical' key with
            per-feature statistics including percentile values.
        production_df: DataFrame of recent inference features.
        features: Subset of features to compute PSI for. Defaults to
            all features present in both reference and production.

    Returns:
        Dictionary mapping feature names to PSI values.
    """
    numerical_stats = reference_stats.get("numerical", {})

    if features is None:
        features = [
            f for f in numerical_stats
            if f in production_df.columns
        ]

    psi_results = {}

    for feature in features:
        if feature not in numerical_stats:
            logger.debug("Feature %s not in reference stats, skipping", feature)
            continue

        if feature not in production_df.columns:
            logger.debug("Feature %s not in production data, skipping", feature)
            continue

        stats = numerical_stats[feature]
        reference_values = _reconstruct_reference_sample(stats)
        production_values = production_df[feature].values

        psi = compute_psi(reference_values, production_values)
        psi_results[feature] = psi

        level = _psi_level(psi)
        logger.info(
            "PSI %s: %.4f (%s)",
            feature,
            psi,
            level,
        )

    return psi_results


def compute_prediction_drift(
    reference_scores: np.ndarray,
    production_scores: np.ndarray,
) -> dict[str, float]:
    """Compute drift metrics on predicted scores.

    Tracks shifts in the score distribution that signal upstream
    data problems without requiring ground truth labels.

    Args:
        reference_scores: Model scores on training/validation data.
        production_scores: Recent predicted scores from inference log.

    Returns:
        Dictionary with PSI and mean shift metrics.
    """
    psi = compute_psi(reference_scores, production_scores)
    mean_shift = float(
        np.mean(production_scores) - np.mean(reference_scores)
    )

    return {
        "score_psi": psi,
        "score_mean_shift": mean_shift,
        "production_mean": float(np.mean(production_scores)),
        "reference_mean": float(np.mean(reference_scores)),
    }


def _reconstruct_reference_sample(stats: dict) -> np.ndarray:
    """Reconstruct an approximate reference distribution from percentiles.

    Since we store percentiles rather than raw values, we reconstruct
    a representative sample by interpolating between stored percentiles.
    This is sufficient for PSI computation which only needs the
    distributional shape, not exact values.

    Args:
        stats: Feature statistics dict with p5, p25, p50, p75, p95 keys.

    Returns:
        Approximate sample from the reference distribution.
    """
    percentile_values = np.array([
        stats["p5"],
        stats["p25"],
        stats["p50"],
        stats["p75"],
        stats["p95"],
    ])
    percentile_points = np.array([5, 25, 50, 75, 95])

    sample_points = np.linspace(5, 95, 1000)
    sample = np.interp(sample_points, percentile_points, percentile_values)

    return sample


def _psi_level(psi: float) -> str:
    """Return a human-readable stability level for a PSI value.

    Args:
        psi: PSI value.

    Returns:
        'stable', 'warning', or 'drift'.
    """
    if psi < PSI_STABLE:
        return "stable"
    if psi < PSI_WARNING:
        return "warning"
    return "drift"