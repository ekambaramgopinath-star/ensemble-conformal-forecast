"""
stats_test.py — Statistical comparison tests for the thesis.

Used in §4.3 and §4.4 to formally test whether differences between
EnbPI and AgACI (or LSTM vs GRU) are statistically significant.

Tests implemented
-----------------
diebold_mariano   — Is forecast A significantly more accurate than B?
                    (§4.2 point forecast comparison)
kupiec_pof_test   — Is the interval failure rate = alpha?
                    (re-exported from evaluation.py for convenience)
compare_asset_coverage — Cross-asset coverage summary (§4.4 / §5.x)
"""

import numpy as np
import pandas as pd
from scipy import stats

from src.evaluation import kupiec_pof_test  # noqa: re-export


# ──────────────────────────────────────────────
#  DIEBOLD-MARIANO TEST
#  §4.2 — Is LSTM significantly better than GRU (or vice versa)?
# ──────────────────────────────────────────────

def diebold_mariano(y_true, pred_a, pred_b, loss="mse", h=1):
    """
    Diebold-Mariano test for equal predictive accuracy.

    H₀: E[d_t] = 0  (models A and B have equal expected loss)
    H₁: E[d_t] ≠ 0  (one model is significantly better)

    Parameters
    ----------
    y_true : array-like   — actual values
    pred_a : array-like   — forecasts from model A
    pred_b : array-like   — forecasts from model B
    loss   : str          — 'mse' (default) or 'mae'
    h      : int          — forecast horizon (1-step default)

    Returns
    -------
    dm_stat : float   — DM test statistic
    p_value : float   — two-sided p-value (< 0.05 → reject H₀)

    Reference: Diebold & Mariano (1995), JBES.
    """
    y = np.asarray(y_true, dtype=float)
    a = np.asarray(pred_a, dtype=float)
    b = np.asarray(pred_b, dtype=float)

    if loss == "mse":
        e_a = (y - a) ** 2
        e_b = (y - b) ** 2
    elif loss == "mae":
        e_a = np.abs(y - a)
        e_b = np.abs(y - b)
    else:
        raise ValueError("loss must be 'mse' or 'mae'")

    d = e_a - e_b
    T = len(d)
    d_bar = d.mean()

    # Newey-West variance with h-1 lags
    gamma_0 = np.var(d, ddof=1)
    nw_var = gamma_0
    for lag in range(1, h):
        gamma_l = np.mean((d[lag:] - d_bar) * (d[:-lag] - d_bar))
        nw_var += 2.0 * (1.0 - lag / h) * gamma_l

    if nw_var <= 0:
        return np.nan, np.nan

    dm_stat = d_bar / np.sqrt(nw_var / T)
    p_value = float(2.0 * (1.0 - stats.norm.cdf(abs(dm_stat))))

    return float(dm_stat), p_value


# ──────────────────────────────────────────────
#  CROSS-ASSET COMPARISON
#  §4.4 / §5.x
# ──────────────────────────────────────────────

def compare_asset_coverage(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate EnbPI and AgACI coverage statistics across asset classes.

    Input DataFrame must have columns:
        asset_type, enbpi_coverage, agaci_coverage, enbpi_width
    """
    return (
        df.groupby("asset_type")
        .agg(
            enbpi_coverage_mean=("enbpi_coverage", "mean"),
            agaci_coverage_mean=("agaci_coverage", "mean"),
            enbpi_width_mean=("enbpi_width", "mean"),
        )
        .reset_index()
    )
