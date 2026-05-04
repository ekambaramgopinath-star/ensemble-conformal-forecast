import numpy as np


# ============================================================
# RESIDUALS (returns space)
# ============================================================
def get_residuals(y_true, y_pred):
    """
    Absolute residuals in return space.
    """
    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()

    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have same length")

    return np.abs(y_true - y_pred)


# Optional: scaled residuals (not used in main pipeline, but useful research tool)
def get_scaled_residuals(y_true, y_pred, std):
    """
    Standardized residuals (for heteroskedastic-aware extensions).
    """
    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()
    std = np.asarray(std).flatten()

    return np.abs(y_true - y_pred) / (std + 1e-8)


# ============================================================
# ENBPI (correct + stable implementation)
# ============================================================
def apply_enbpi(test_preds, cal_residuals, alpha=0.1):
    """
    Ensemble Batch Prediction Intervals (EnbPI)
    """
    test_preds = np.asarray(test_preds).flatten()
    cal_residuals = np.asarray(cal_residuals).flatten()

    if len(cal_residuals) == 0:
        raise ValueError("Calibration residuals cannot be empty")

    n = len(cal_residuals)

    # Finite-sample corrected quantile
    q_level = np.ceil((n + 1) * (1 - alpha)) / n
    q_level = min(q_level, 1.0)

    q = np.quantile(cal_residuals, q_level)

    lower = test_preds - q
    upper = test_preds + q

    return lower, upper


# ============================================================
# AGACI (Improved + stable + thesis-ready)
# ============================================================
def rolling_agaci(
    test_preds,
    y_true,
    cal_residuals,
    alpha=0.1,
    window=50,
    decay=0.9,
    clip_percentile=99
):
    """
    Adaptive conformal inference with:
    - rolling window
    - exponential decay weighting
    - optional residual clipping for robustness
    """

    test_preds = np.asarray(test_preds).flatten()
    y_true = np.asarray(y_true).flatten()

    history = list(np.asarray(cal_residuals).flatten())

    lower, upper = [], []

    for i in range(len(test_preds)):

        # -----------------------------
        # Rolling window
        # -----------------------------
        relevant = np.array(history[-window:])

        if len(relevant) == 0:
            raise ValueError("Not enough residuals for AgACI")

        # -----------------------------
        # Optional robustness: clip extreme residuals
        # (VERY useful for BTC / high volatility assets)
        # -----------------------------
        if clip_percentile is not None:
            cap = np.percentile(relevant, clip_percentile)
            relevant = np.clip(relevant, 0, cap)

        # -----------------------------
        # Exponential decay weights
        # -----------------------------
        steps = np.arange(len(relevant))[::-1]
        weights = np.exp(- (1 - decay) * steps)
        weights = weights / np.sum(weights)

        # -----------------------------
        # Weighted quantile computation
        # -----------------------------
        sorted_idx = np.argsort(relevant)
        sorted_res = relevant[sorted_idx]
        sorted_weights = weights[sorted_idx]

        cumulative = np.cumsum(sorted_weights)

        idx = np.searchsorted(cumulative, 1 - alpha, side="left")
        idx = min(idx, len(sorted_res) - 1)

        q = sorted_res[idx]

        # -----------------------------
        # Prediction interval
        # -----------------------------
        lower.append(test_preds[i] - q)
        upper.append(test_preds[i] + q)

        # -----------------------------
        # Update history with new residual
        # -----------------------------
        new_residual = np.abs(y_true[i] - test_preds[i])
        history.append(new_residual)

    return np.array(lower), np.array(upper)