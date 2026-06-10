import logging
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    brier_score_loss,
    confusion_matrix,
    roc_auc_score,
)

logger = logging.getLogger(__name__)

PLOTS_DIR = Path("ml/evaluation")


def run_evaluation(
    y_val: pd.Series,
    y_pred_proba: np.ndarray,
    output_dir: Optional[Path] = None,
) -> dict[str, float]:
    """Run the full evaluation suite on validation predictions.

    Computes metrics and saves plots to output_dir.

    Args:
        y_val: True labels from the validation set.
        y_pred_proba: Predicted probabilities from the model.
        output_dir: Directory to write plot files. Defaults to PLOTS_DIR.

    Returns:
        Dictionary of all computed metrics.
    """
    output_dir = output_dir or PLOTS_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    metrics = {}

    metrics.update(_compute_core_metrics(y_val, y_pred_proba))
    metrics.update(_compute_threshold_metrics(y_val, y_pred_proba))

    _plot_calibration_curve(y_val, y_pred_proba, output_dir)
    _plot_ks_curve(y_val, y_pred_proba, output_dir)
    _plot_score_distribution(y_val, y_pred_proba, output_dir)
    _plot_decile_analysis(y_val, y_pred_proba, output_dir)

    _log_metrics(metrics)

    return metrics


def _compute_core_metrics(
    y_val: pd.Series,
    y_pred_proba: np.ndarray,
) -> dict[str, float]:
    """Compute AUC, KS statistic, and Brier score.

    Args:
        y_val: True labels.
        y_pred_proba: Predicted probabilities.

    Returns:
        Dictionary of core metrics.
    """
    auc = roc_auc_score(y_val, y_pred_proba)

    default_scores = y_pred_proba[y_val == 1]
    non_default_scores = y_pred_proba[y_val == 0]
    ks_stat, _ = ks_2samp(default_scores, non_default_scores)

    brier = brier_score_loss(y_val, y_pred_proba)

    return {
        "auc_roc": auc,
        "ks_statistic": ks_stat,
        "brier_score": brier,
    }


def _compute_threshold_metrics(
    y_val: pd.Series,
    y_pred_proba: np.ndarray,
    thresholds: Optional[list[float]] = None,
) -> dict[str, float]:
    """Compute approval rate and default rate at business thresholds.

    In credit risk, threshold selection is a business decision, 
    a lower threshold approves more applicants but accepts more defaults.
    This function makes that tradeoff explicit.

    Args:
        y_val: True labels.
        y_pred_proba: Predicted probabilities.
        thresholds: Score thresholds to evaluate. Defaults to
            [0.1, 0.2, 0.3, 0.4, 0.5].

    Returns:
        Dictionary of threshold-specific metrics flattened for MLflow.
    """
    thresholds = thresholds or [0.1, 0.2, 0.3, 0.4, 0.5]
    metrics = {}

    for threshold in thresholds:
        approved = y_pred_proba < threshold
        approval_rate = approved.mean()

        if approved.sum() > 0:
            default_rate = y_val[approved].mean()
        else:
            default_rate = 0.0

        key = str(threshold).replace(".", "_")
        metrics[f"approval_rate_at_{key}"] = approval_rate
        metrics[f"default_rate_at_{key}"] = default_rate

    return metrics


def _plot_calibration_curve(
    y_val: pd.Series,
    y_pred_proba: np.ndarray,
    output_dir: Path,
) -> None:
    """Plot the calibration curve (reliability diagram).

    A well-calibrated model has a curve close to the diagonal, 
    a predicted probability of 0.2 should correspond to a 20%
    observed default rate. In credit risk, calibration matters
    for regulatory compliance and interest rate setting.

    Args:
        y_val: True labels.
        y_pred_proba: Predicted probabilities.
        output_dir: Directory to write the plot.
    """
    fraction_of_positives, mean_predicted_value = calibration_curve(
        y_val, y_pred_proba, n_bins=10
    )

    fig, ax = plt.subplots(figsize=(8, 6))

    ax.plot(
        mean_predicted_value,
        fraction_of_positives,
        "s-",
        label="LightGBM",
        color="#1f77b4",
    )
    ax.plot(
        [0, 1], [0, 1],
        "k--",
        label="Perfectly calibrated",
    )

    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Fraction of positives")
    ax.set_title("Calibration curve")
    ax.legend()
    ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_dir / "calibration_curve.png", dpi=150)
    plt.close(fig)

    logger.info("Calibration curve saved")


def _plot_ks_curve(
    y_val: pd.Series,
    y_pred_proba: np.ndarray,
    output_dir: Path,
) -> None:
    """Plot the KS curve showing separation between score distributions.

    The KS statistic is the maximum vertical distance between the
    cumulative distribution functions of default and non-default scores.
    It identifies the threshold with maximum discriminatory power.

    Args:
        y_val: True labels.
        y_pred_proba: Predicted probabilities.
        output_dir: Directory to write the plot.
    """
    df = pd.DataFrame({"score": y_pred_proba, "target": y_val})
    df = df.sort_values("score")

    defaults = df[df["target"] == 1]["score"]
    non_defaults = df[df["target"] == 0]["score"]

    thresholds = np.linspace(0, 1, 200)
    cdf_default = [(defaults <= t).mean() for t in thresholds]
    cdf_non_default = [(non_defaults <= t).mean() for t in thresholds]
    ks_values = np.abs(np.array(cdf_default) - np.array(cdf_non_default))
    ks_idx = np.argmax(ks_values)

    fig, ax = plt.subplots(figsize=(8, 6))

    ax.plot(thresholds, cdf_default, label="Defaults", color="#d62728")
    ax.plot(thresholds, cdf_non_default, label="Non-defaults", color="#1f77b4")
    ax.axvline(
        thresholds[ks_idx],
        color="gray",
        linestyle="--",
        label=f"KS = {ks_values[ks_idx]:.3f}",
    )

    ax.set_xlabel("Score threshold")
    ax.set_ylabel("Cumulative fraction")
    ax.set_title("KS curve")
    ax.legend()
    ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_dir / "ks_curve.png", dpi=150)
    plt.close(fig)

    logger.info("KS curve saved")


def _plot_score_distribution(
    y_val: pd.Series,
    y_pred_proba: np.ndarray,
    output_dir: Path,
) -> None:
    """Plot score distributions for defaults and non-defaults.

    Visual separation between the two distributions is the intuition
    behind AUC and KS, a good model pushes defaults to the right
    and non-defaults to the left.

    Args:
        y_val: True labels.
        y_pred_proba: Predicted probabilities.
        output_dir: Directory to write the plot.
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    ax.hist(
        y_pred_proba[y_val == 0],
        bins=50,
        alpha=0.6,
        density=True,
        label="Non-default",
        color="#1f77b4",
    )
    ax.hist(
        y_pred_proba[y_val == 1],
        bins=50,
        alpha=0.6,
        density=True,
        label="Default",
        color="#d62728",
    )

    ax.set_xlabel("Predicted probability of default")
    ax.set_ylabel("Density")
    ax.set_title("Score distribution by outcome")
    ax.legend()
    ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_dir / "score_distribution.png", dpi=150)
    plt.close(fig)

    logger.info("Score distribution saved")


def _plot_decile_analysis(
    y_val: pd.Series,
    y_pred_proba: np.ndarray,
    output_dir: Path,
) -> None:
    """Plot default rate by score decile.

    Decile analysis is the standard credit risk presentation format.
    A well-ordered model shows monotonically increasing default rates
    from the lowest-risk decile to the highest-risk decile.

    Args:
        y_val: True labels.
        y_pred_proba: Predicted probabilities.
        output_dir: Directory to write the plot.
    """
    df = pd.DataFrame({"score": y_pred_proba, "target": y_val})
    df["decile"] = pd.qcut(df["score"], q=10, labels=False)

    decile_stats = df.groupby("decile").agg(
        default_rate=("target", "mean"),
        count=("target", "count"),
        mean_score=("score", "mean"),
    ).reset_index()

    fig, ax = plt.subplots(figsize=(10, 6))

    bars = ax.bar(
        decile_stats["decile"] + 1,
        decile_stats["default_rate"],
        color="#1f77b4",
        alpha=0.8,
        edgecolor="white",
    )

    ax.set_xlabel("Score decile (1 = lowest risk, 10 = highest risk)")
    ax.set_ylabel("Default rate")
    ax.set_title("Default rate by score decile")
    ax.set_xticks(range(1, 11))
    ax.grid(axis="y", alpha=0.3)

    for bar, rate in zip(bars, decile_stats["default_rate"]):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.002,
            f"{rate:.1%}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    fig.tight_layout()
    fig.savefig(output_dir / "decile_analysis.png", dpi=150)
    plt.close(fig)

    logger.info("Decile analysis saved")


def _log_metrics(metrics: dict[str, float]) -> None:
    """Log all metrics to the console.

    Args:
        metrics: Dictionary of metric names to values.
    """
    logger.info("Evaluation results:")
    for name, value in sorted(metrics.items()):
        logger.info("  %s: %.4f", name, value)