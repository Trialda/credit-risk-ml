import logging

import numpy as np
import pytest

from app.services.drift import PSI_STABLE, PSI_WARNING, compute_psi

logger = logging.getLogger(__name__)


def test_psi_identical_distributions_returns_zero() -> None:
    """PSI between identical distributions should be approximately zero."""
    reference = np.random.normal(0, 1, 1000)
    production = reference.copy()
    psi = compute_psi(reference, production)
    assert psi < PSI_STABLE


def test_psi_similar_distributions_is_stable() -> None:
    """PSI between similar distributions should be below warning threshold."""
    np.random.seed(42)
    reference = np.random.normal(0, 1, 1000)
    production = np.random.normal(0.05, 1.05, 1000)
    psi = compute_psi(reference, production)
    assert psi < PSI_STABLE


def test_psi_shifted_distribution_exceeds_warning() -> None:
    """PSI between significantly shifted distributions should exceed warning."""
    np.random.seed(42)
    reference = np.random.normal(0, 1, 1000)
    production = np.random.normal(3, 1, 1000)
    psi = compute_psi(reference, production)
    assert psi >= PSI_WARNING


def test_psi_severely_drifted_distribution_exceeds_drift_threshold() -> None:
    """PSI between severely drifted distributions should exceed drift threshold."""
    np.random.seed(42)
    reference = np.random.normal(0, 1, 1000)
    production = np.random.normal(5, 1, 1000)
    psi = compute_psi(reference, production)
    assert psi >= PSI_WARNING


def test_psi_empty_reference_returns_zero() -> None:
    """PSI with empty reference array returns 0 without crashing."""
    reference = np.array([])
    production = np.random.normal(0, 1, 100)
    psi = compute_psi(reference, production)
    assert psi == 0.0


def test_psi_empty_production_returns_zero() -> None:
    """PSI with empty production array returns 0 without crashing."""
    reference = np.random.normal(0, 1, 1000)
    production = np.array([])
    psi = compute_psi(reference, production)
    assert psi == 0.0


def test_psi_constant_feature_returns_zero() -> None:
    """PSI for a constant feature (no variance) returns 0 without crashing."""
    reference = np.ones(1000)
    production = np.ones(500)
    psi = compute_psi(reference, production)
    assert psi == 0.0


def test_psi_is_non_negative() -> None:
    """PSI should always be non-negative."""
    np.random.seed(42)
    reference = np.random.normal(0, 1, 1000)
    production = np.random.normal(1, 2, 1000)
    psi = compute_psi(reference, production)
    assert psi >= 0.0


def test_psi_increases_with_drift_magnitude() -> None:
    """PSI should increase as the distribution shift increases."""
    np.random.seed(42)
    reference = np.random.normal(0, 1, 1000)
    psi_small = compute_psi(reference, np.random.normal(1, 1, 1000))
    psi_large = compute_psi(reference, np.random.normal(3, 1, 1000))
    assert psi_large > psi_small