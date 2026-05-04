"""
evaluation.py — Thesis-aligned metrics for "Estimating forecast uncertainty
in financial time series".

Chapter mapping
---------------
§4.2  Point forecast accuracy : rmse, mae, mape
§4.3  Interval quality         : coverage, interval_width, winkler_score,
                                  coverage_deviation, kupiec_pof_test
§4.4  Adaptive behaviour       : conditional_coverage
§4.5  (sensitivity notebook uses these same helpers)
§5.x  Practical utility        : sharpe  (also used by backtest.py)
Statistical comparison          : diebold_mariano  (stats_test.py)
"""

import numpy as np
from scipy import stats
from sklearn.metrics import mean_squared_error


# ──────────────────────────────────────────────
#  §4.2  POINT FORECAST ACCURACY
# ──────────────────────────────────────────────

def rmse(y_true, y_pred):
    """Root Mean Squared Error — penalises large errors."""
    return np.sqrt(mean_squared_error(y_true, y_pred))


def mae(y_true, y_pred):
    """Mean Absolute Error — robust companion to RMSE."""
    return np.mean(np.abs(np.asarray(y_true) - np.asarray(y_pred)))


def mape(y_true, y_pred):
    """Mean Absolute Percentage Error (%)."""
    y_true = np.asarray(y_true, dtype=float)
    return np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100


# ──────────────────────────────────────────────
#  §4.3  INTERVAL QUALITY
# ──────────────────────────────────────────────

def coverage(y_true, lower, upper):
    """Empirical coverage: fraction of actuals inside the interval."""
    y_arr = np.asarray(y_true, dtype=float).reshape(-1)
    lower_arr = np.asarray(lower, dtype=float).reshape(-1)
    upper_arr = np.asarray(upper, dtype=float).reshape(-1)

    if not (len(y_arr) == len(lower_arr) == len(upper_arr)):
        raise ValueError(
            "coverage inputs must have equal length after flattening. "
            f"Got y_true={len(y_arr)}, lower={len(lower_arr)}, "
            f"upper={len(upper_arr)}"
        )

    return np.mean((y_arr >= lower_arr) & (y_arr <= upper_arr))


def interval_width(lower, upper):
    """Average prediction interval width — measures efficiency."""
    return np.mean(np.asarray(upper) - np.asarray(lower))


def coverage_deviation(y_true, lower, upper, alpha):
    """
    |empirical coverage − nominal coverage|.
    A well-calibrated method should be close to zero.
    """
    target = 1.0 - alpha
    return abs(coverage(y_true, lower, upper) - target)


def winkler_score(y_true, lower, upper, alpha):
    """
    Winkler Interval Score (proper scoring rule for prediction intervals).
    Lower is better.  Penalises both wide intervals and coverage failures.

    Formula: (u - l) + (2/alpha)*max(l - y, 0) + (2/alpha)*max(y - u, 0)
    Reference: Winkler (1972); Gneiting & Raftery (2007).
    """
    y_arr = np.asarray(y_true, dtype=float)
    lower_arr = np.asarray(lower, dtype=float)
    upper_arr = np.asarray(upper, dtype=float)

    width = upper_arr - lower_arr
    penalty_low = (2.0 / alpha) * np.maximum(lower_arr - y_arr, 0.0)
    penalty_high = (2.0 / alpha) * np.maximum(y_arr - upper_arr, 0.0)
    return float(np.mean(width + penalty_low + penalty_high))


def kupiec_pof_test(y_true, lower, upper, alpha):
    """
    Kupiec Proportion of Failures (POF) test.
    H₀: empirical failure rate = alpha  (interval is correctly calibrated).

    Returns
    -------
    lr_stat : float   — likelihood-ratio test statistic (chi-squared)
    p_value : float   — p > 0.05 → fail to reject H₀ (good calibration)

    Reference: Kupiec (1995), Journal of Derivatives.
    """
    y_arr = np.asarray(y_true, dtype=float)
    lower_arr = np.asarray(lower, dtype=float)
    upper_arr = np.asarray(upper, dtype=float)

    T = len(y_arr)
    violations = np.sum(
        (y_arr < lower_arr) | (y_arr > upper_arr)
    )  # failures (not covered)
    x = int(violations)

    if x == 0 or x == T:
        return np.nan, np.nan

    p_hat = x / T  # observed failure rate

    # Log-likelihood ratio: LR_POF ~ chi-squared(1) under H₀
    lr_stat = -2.0 * (
        x * np.log(alpha / p_hat) + (T - x) * np.log((1 - alpha) / (1 - p_hat))
    )
    p_value = float(1.0 - stats.chi2.cdf(lr_stat, df=1))

    return float(lr_stat), p_value


# ──────────────────────────────────────────────
#  §4.4  ADAPTIVE BEHAVIOUR
# ──────────────────────────────────────────────

def conditional_coverage(y_true, lower, upper, returns, quantile=0.75):
    """
    Compare coverage in HIGH-volatility vs LOW-volatility periods.

    Splits the test set by absolute daily return magnitude:
      - high-vol : |return| >= quantile threshold
      - low-vol  : |return| <  quantile threshold

    Returns a dict with coverage for each regime.
    This is the key evaluation for §4.4 (adaptive behaviour).
    """
    y_arr = np.asarray(y_true, dtype=float).reshape(-1)
    lower_arr = np.asarray(lower, dtype=float).reshape(-1)
    upper_arr = np.asarray(upper, dtype=float).reshape(-1)
    returns_arr = np.asarray(returns, dtype=float).reshape(-1)

    if not (len(y_arr) == len(lower_arr) == len(upper_arr)):
        raise ValueError(
            "conditional_coverage interval inputs must have equal length "
            f"after flattening. Got y_true={len(y_arr)}, "
            f"lower={len(lower_arr)}, upper={len(upper_arr)}"
        )

    if len(returns_arr) < len(y_arr):
        returns_arr = np.append(0.0, returns_arr)
    abs_r = np.abs(returns_arr[: len(y_arr)])

    threshold = np.quantile(abs_r, quantile)
    high_vol = abs_r >= threshold
    low_vol = ~high_vol

    covered = (y_arr >= lower_arr) & (y_arr <= upper_arr)

    return {
        "high_vol_coverage": (
            float(np.mean(covered[high_vol])) if high_vol.any() else np.nan
        ),
        "low_vol_coverage": (
            float(np.mean(covered[low_vol])) if low_vol.any() else np.nan
        ),
        "high_vol_n": int(high_vol.sum()),
        "low_vol_n": int(low_vol.sum()),
    }


# ──────────────────────────────────────────────
#  §5.x  PRACTICAL UTILITY
# ──────────────────────────────────────────────

def sharpe(returns):
    """Annualised Sharpe Ratio (risk-free rate = 0)."""
    returns = np.asarray(returns)
    if returns.std() == 0:
        return 0.0
    return float(returns.mean() / returns.std() * np.sqrt(252))
