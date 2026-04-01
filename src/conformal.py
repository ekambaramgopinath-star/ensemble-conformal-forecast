import numpy as np


def get_residuals(y_true, y_pred):
    return np.abs(y_true.flatten() - y_pred.flatten())


def apply_enbpi(test_preds, cal_residuals, alpha=0.1):
    q = np.quantile(cal_residuals, 1 - alpha)
    return test_preds - q, test_preds + q


def rolling_agaci(test_preds, y_true, cal_residuals, alpha=0.1, window=50):
    n = len(test_preds)
    lower, upper = [], []
    history = list(cal_residuals)

    y_true_flat = np.asarray(y_true).flatten()
    for i in range(n):
        relevant_res = history[-window:]
        q = np.quantile(relevant_res, 1 - alpha)
        lower.append(test_preds[i] - q)
        upper.append(test_preds[i] + q)

        observed_error = abs(y_true_flat[i] - test_preds[i])
        history.append(observed_error)

    return np.array(lower), np.array(upper)
