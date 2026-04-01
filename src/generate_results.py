import os
import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

from src.visualization import plot_rolling_coverage


# ------------------------------
# 1. Helper Functions
# ------------------------------

def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))


def mape(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


def interval_width(lower, upper):
    return np.mean(upper - lower)


def coverage(y_true, lower, upper):
    return np.mean((y_true >= lower) & (y_true <= upper))


def main():
    config_path = os.path.join(
        os.path.dirname(__file__),
        "..",
        "configs",
        "config.yaml",
    )
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    alpha = config["conformal"]["alpha"]

    results_dir = os.path.join(
        os.path.dirname(__file__),
        "..",
        "results",
    )
    reports_dir = os.path.join(
        os.path.dirname(__file__),
        "..",
        "reports",
    )
    os.makedirs(reports_dir, exist_ok=True)

    result_files = [
        os.path.join(results_dir, filename)
        for filename in os.listdir(results_dir)
        if filename.endswith(".csv")
    ]
    if not result_files:
        print("No result files found in /results. Run run_pipeline.py first.")
        return

    latest_file = max(result_files, key=os.path.getctime)
    print(f"Generating Tables & Plots from: {latest_file}")

    df = pd.read_csv(latest_file)

    y_test = np.asarray(df["Actual"])
    models_data = {
        "LSTM": {
            "pred": np.asarray(df["LSTM_Pred"]),
            "l_e": np.asarray(df["LSTM_EnbPI_Lower"]),
            "u_e": np.asarray(df["LSTM_EnbPI_Upper"]),
            "l_a": np.asarray(df["LSTM_AgACI_Lower"]),
            "u_a": np.asarray(df["LSTM_AgACI_Upper"]),
        },
        "GRU": {
            "pred": np.asarray(df["GRU_Pred"]),
            "l_e": np.asarray(df["GRU_EnbPI_Lower"]),
            "u_e": np.asarray(df["GRU_EnbPI_Upper"]),
            "l_a": np.asarray(df["GRU_AgACI_Lower"]),
            "u_a": np.asarray(df["GRU_AgACI_Upper"]),
        },
    }

    baseline_pred = None
    if "Baseline_Pred" in df.columns:
        baseline_pred = np.asarray(df["Baseline_Pred"])

    # ------------------------------
    # 2. Table 4.2: Point Forecast Accuracy
    # ------------------------------
    accuracy_results = []
    for m_name, m_val in models_data.items():
        accuracy_results.append(
            {
                "Model": m_name,
                "RMSE": rmse(y_test, m_val["pred"]),
                "MAPE (%)": mape(y_test, m_val["pred"]),
            }
        )

    if baseline_pred is not None:
        accuracy_results.append(
            {
                "Model": "Baseline",
                "RMSE": rmse(y_test, baseline_pred),
                "MAPE (%)": mape(y_test, baseline_pred),
            }
        )

    df_accuracy = pd.DataFrame(accuracy_results)
    df_accuracy.to_csv(
        os.path.join(reports_dir, "table_4_2_point_accuracy.csv"),
        index=False,
    )
    print("\n--- Table 4.2 Saved ---")
    print(df_accuracy)

    # ------------------------------
    # 3. Table 4.3: Prediction Interval Evaluation
    # ------------------------------
    interval_results = []
    for m_name, m_val in models_data.items():
        interval_results.append(
            {
                "Model": m_name,
                "Method": "EnbPI",
                "Coverage": coverage(y_test, m_val["l_e"], m_val["u_e"]),
                "Target": 1 - alpha,
                "Avg Width": interval_width(m_val["l_e"], m_val["u_e"]),
            }
        )
        interval_results.append(
            {
                "Model": m_name,
                "Method": "AgACI",
                "Coverage": coverage(y_test, m_val["l_a"], m_val["u_a"]),
                "Target": 1 - alpha,
                "Avg Width": interval_width(m_val["l_a"], m_val["u_a"]),
            }
        )

    df_intervals = pd.DataFrame(interval_results)
    df_intervals.to_csv(
        os.path.join(reports_dir, "table_4_3_interval_metrics.csv"),
        index=False,
    )
    print("\n--- Table 4.3 Saved ---")
    print(df_intervals)

    # ------------------------------
    # 4. Plots (Forecasts & Rolling Coverage)
    # ------------------------------
    for m_name, m_val in models_data.items():
        plt.figure(figsize=(12, 5))
        plt.plot(y_test, label="Actual", color="black", alpha=0.6)
        plt.plot(m_val["pred"], label=f"{m_name} Forecast", color="blue")
        plt.fill_between(
            range(len(y_test)),
            m_val["l_e"],
            m_val["u_e"],
            alpha=0.2,
            color="blue",
            label="EnbPI 90% PI",
        )
        plt.fill_between(
            range(len(y_test)),
            m_val["l_a"],
            m_val["u_a"],
            alpha=0.2,
            color="red",
            label="AgACI 90% PI",
        )
        plt.title(
            f"Figure 4.1: {m_name} Forecast vs Adaptive Intervals"
        )
        plt.legend()
        plt.savefig(
            os.path.join(reports_dir, f"{m_name}_intervals.png"),
            dpi=300,
        )
        plt.close()

        plot_rolling_coverage(
            y_test,
            m_val["l_e"],
            m_val["u_e"],
            m_val["l_a"],
            m_val["u_a"],
            model_name=m_name,
            window=20,
            save_path=os.path.join(
                reports_dir,
                f"{m_name}_rolling_coverage.png",
            ),
        )

    print(f"\nAll Chapter 4 Artifacts generated in: {reports_dir}")


if __name__ == "__main__":
    main()
