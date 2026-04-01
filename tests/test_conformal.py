import numpy as np
from src.conformal import get_residuals, apply_enbpi, rolling_agaci


def test_get_residuals_and_apply_enbpi():
    y_true = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([1.1, 1.8, 2.5])

    residuals = get_residuals(y_true, y_pred)
    assert np.allclose(residuals, np.array([0.1, 0.2, 0.5]))

    lower, upper = apply_enbpi(np.array([2.5, 3.0]), residuals, alpha=0.2)
    q = np.quantile(residuals, 1 - 0.2)
    assert lower.shape == (2,)
    assert upper.shape == (2,)
    assert np.allclose(lower, np.array([2.5 - q, 3.0 - q]))
    assert np.allclose(upper, np.array([2.5 + q, 3.0 + q]))


def test_rolling_agaci_adapts_to_new_errors():
    y_true = np.array([1.0, 2.0, 3.0, 4.0])
    preds = np.array([1.0, 2.0, 2.0, 5.0])
    cal_res = np.array([0.2, 0.1, 0.3])

    lower, upper = rolling_agaci(
        preds,
        y_true,
        cal_res,
        alpha=0.2,
        window=3,
    )

    assert lower.shape == preds.shape
    assert upper.shape == preds.shape
    assert np.all(lower <= preds)
    assert np.all(upper >= preds)
