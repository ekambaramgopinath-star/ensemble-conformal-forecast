import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_forecast_vs_actual(y_test, lstm_pred, gru_pred):
    """Plot 1 — Comparison of point forecasts."""
    plt.figure(figsize=(12, 5))
    plt.plot(y_test, label="Actual", color="black", linewidth=1.5)
    plt.plot(lstm_pred, label="LSTM Forecast", color="blue", linestyle="--")
    plt.plot(gru_pred, label="GRU Forecast", color="green", linestyle="--")
    plt.title("Point Forecast vs Actual Price")
    plt.xlabel("Time Steps")
    plt.ylabel("Normalized Price")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.show()


def plot_prediction_intervals(
    y_test,
    pred,
    lower_enbpi,
    upper_enbpi,
    lower_agaci,
    upper_agaci,
    model_name="LSTM",
):
    """Plot 2 — Visual comparison of EnbPI vs AgACI."""
    plt.figure(figsize=(12, 5))
    plt.plot(y_test, label="Actual", color="black", alpha=0.7)
    plt.plot(pred, label=f"{model_name} Forecast", color="blue")

    plt.fill_between(
        range(len(lower_enbpi)),
        lower_enbpi,
        upper_enbpi,
        alpha=0.2,
        color="blue",
        label=f"{model_name} EnbPI 90% PI",
    )

    plt.fill_between(
        range(len(lower_agaci)),
        lower_agaci,
        upper_agaci,
        alpha=0.3,
        color="red",
        label=f"{model_name} AgACI 90% PI",
    )

    plt.title(f"{model_name} Prediction Intervals: EnbPI vs AgACI")
    plt.xlabel("Time Steps")
    plt.ylabel("Normalized Price")
    plt.legend()
    plt.show()


def calculate_metrics(y_true, lower, upper):
    """Plot 3 — Coverage and Width Metrics."""
    y_true = y_true.flatten()
    covered = (y_true >= lower.flatten()) & (y_true <= upper.flatten())
    coverage = np.mean(covered)
    avg_width = np.mean(upper - lower)
    return coverage, avg_width


def plot_rolling_coverage(
    y_true,
    lower_e,
    upper_e,
    lower_a,
    upper_a,
    model_name="LSTM",
    window=20,
    save_path=None,
):
    """Compute and plot the rolling coverage mean."""
    cov_e = ((y_true >= lower_e) & (y_true <= upper_e)).astype(float)
    cov_a = ((y_true >= lower_a) & (y_true <= upper_a)).astype(float)

    roll_e = pd.Series(cov_e).rolling(window=window).mean()
    roll_a = pd.Series(cov_a).rolling(window=window).mean()

    plt.figure(figsize=(12, 5))
    plt.axhline(
        y=0.90,
        color="black",
        linestyle="--",
        label="Target Coverage (90%)",
    )

    plt.plot(
        roll_e,
        label=f"{model_name} EnbPI (Static)",
        color="blue",
        alpha=0.5,
    )
    plt.plot(
        roll_a,
        label=f"{model_name} AgACI (Adaptive)",
        color="red",
        linewidth=2,
    )

    plt.title(
        f"Local Coverage Reliability: {model_name} ({window}-day Rolling)"
    )
    plt.xlabel("Time Steps")
    plt.ylabel("Coverage Probability")
    plt.ylim(0.5, 1.05)
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)

    if save_path:
        plt.savefig(save_path, dpi=300)

    plt.show()
