"""
generate_results.py — Chapter 4 artefact generator.

Run after run_pipeline.py to produce all thesis tables and figures.

Chapter 4 mapping
-----------------
§4.2  Point Forecast Accuracy    → reports/table_4_2_point_accuracy.csv
§4.3  Prediction Interval Quality → reports/table_4_3_interval_metrics.csv
§4.4  Adaptive Behaviour          → reports/table_4_4_conditional_coverage.csv
                                    reports/{MODEL}_intervals.png
                                    reports/{MODEL}_rolling_coverage.png
§4.5  (Sensitivity analysis lives in notebooks/sensitivity_analysis.ipynb)
"""

import os
import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.evaluation import (
    rmse,
    mae,
    mape,
    coverage,
    interval_width,
    coverage_deviation,
    winkler_score,
    kupiec_pof_test,
    conditional_coverage,
)
from src.stats_test import diebold_mariano
from src.visualization import plot_rolling_coverage


def main():
    config_path = os.path.join(
        os.path.dirname(__file__), "..", "configs", "config.yaml"
    )
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    alpha = config["conformal"]["alpha"]

    results_dir = os.path.join(os.path.dirname(__file__), "..", "results")
    reports_dir = os.path.join(os.path.dirname(__file__), "..", "reports")
    os.makedirs(reports_dir, exist_ok=True)

    result_files = [
        os.path.join(results_dir, fn)
        for fn in os.listdir(results_dir)
        if fn.endswith(".csv")
    ]
    if not result_files:
        print("No result files found in /results. Run run_pipeline.py first.")
        return

    latest_file = max(result_files, key=os.path.getctime)
    print(f"Generating Chapter 4 artefacts from: {latest_file}\n")

    df = pd.read_csv(latest_file)
    y_test = df["Actual"].to_numpy(dtype=float)

    # Daily returns for conditional coverage (§4.4)
    returns = np.diff(y_test) / (np.abs(y_test[:-1]) + 1e-8)

    models_data = {
        "LSTM": {
            "pred": df["LSTM_Pred"].to_numpy(float),
            "l_e": df["LSTM_EnbPI_Lower"].to_numpy(float),
            "u_e": df["LSTM_EnbPI_Upper"].to_numpy(float),
            "l_a": df["LSTM_AgACI_Lower"].to_numpy(float),
            "u_a": df["LSTM_AgACI_Upper"].to_numpy(float),
        },
        "GRU": {
            "pred": df["GRU_Pred"].to_numpy(float),
            "l_e": df["GRU_EnbPI_Lower"].to_numpy(float),
            "u_e": df["GRU_EnbPI_Upper"].to_numpy(float),
            "l_a": df["GRU_AgACI_Lower"].to_numpy(float),
            "u_a": df["GRU_AgACI_Upper"].to_numpy(float),
        },
    }

    baseline_pred = (
        df["Baseline_Pred"].to_numpy(float)
        if "Baseline_Pred" in df.columns
        else None
    )

    # ──────────────────────────────────────────────────────────
    #  §4.2  TABLE 4.2 — Point Forecast Accuracy
    # ──────────────────────────────────────────────────────────
    accuracy_rows = []
    for m_name, m in models_data.items():
        accuracy_rows.append(
            {
                "Model": m_name,
                "RMSE": round(rmse(y_test, m["pred"]), 4),
                "MAE": round(mae(y_test, m["pred"]), 4),
                "MAPE (%)": round(mape(y_test, m["pred"]), 2),
            }
        )
    if baseline_pred is not None:
        accuracy_rows.append(
            {
                "Model": "Baseline (Persistence)",
                "RMSE": round(rmse(y_test, baseline_pred), 4),
                "MAE": round(mae(y_test, baseline_pred), 4),
                "MAPE (%)": round(mape(y_test, baseline_pred), 2),
            }
        )

    # Diebold-Mariano: LSTM vs GRU significance
    dm_stat, dm_p = diebold_mariano(
        y_test, models_data["LSTM"]["pred"], models_data["GRU"]["pred"]
    )

    df_accuracy = pd.DataFrame(accuracy_rows)
    df_accuracy.to_csv(
        os.path.join(reports_dir, "table_4_2_point_accuracy.csv"), index=False
    )
    print("── §4.2  Table 4.2: Point Forecast Accuracy ──")
    print(df_accuracy.to_string(index=False))
    print(
        f"\n  Diebold-Mariano (LSTM vs GRU): stat={dm_stat:.4f}, p={dm_p:.4f}"
        + ("  ← significant" if pd.notna(dm_p) and dm_p < 0.05 else "  ← not significant")
    )

    # ──────────────────────────────────────────────────────────
    #  §4.3  TABLE 4.3 — Prediction Interval Quality
    # ──────────────────────────────────────────────────────────
    interval_rows = []
    for m_name, m in models_data.items():
        for method_label, l, u in [
            ("EnbPI", m["l_e"], m["u_e"]),
            ("AgACI", m["l_a"], m["u_a"]),
        ]:
            lr, pval = kupiec_pof_test(y_test, l, u, alpha)
            interval_rows.append(
                {
                    "Model": m_name,
                    "Method": method_label,
                    "Target Coverage": f"{(1-alpha):.0%}",
                    "Empirical Coverage": round(coverage(y_test, l, u), 4),
                    "Coverage Deviation": round(coverage_deviation(y_test, l, u, alpha), 4),
                    "Avg Width": round(interval_width(l, u), 4),
                    "Winkler Score": round(winkler_score(y_test, l, u, alpha), 4),
                    "Kupiec p-value": round(pval, 4) if pd.notna(pval) else "N/A",
                }
            )

    df_intervals = pd.DataFrame(interval_rows)
    df_intervals.to_csv(
        os.path.join(reports_dir, "table_4_3_interval_metrics.csv"), index=False
    )
    print("\n── §4.3  Table 4.3: Prediction Interval Quality ──")
    print(df_intervals.to_string(index=False))
    print(
        "\n  Note: Kupiec p > 0.05 → interval is well-calibrated (fail to reject H₀)"
    )

    # ──────────────────────────────────────────────────────────
    #  §4.4  TABLE 4.4 — Conditional Coverage (Volatile vs Calm)
    # ──────────────────────────────────────────────────────────
    cond_rows = []
    for m_name, m in models_data.items():
        for method_label, l, u in [
            ("EnbPI", m["l_e"], m["u_e"]),
            ("AgACI", m["l_a"], m["u_a"]),
        ]:
            cc = conditional_coverage(y_test, l, u, returns)
            cond_rows.append(
                {
                    "Model": m_name,
                    "Method": method_label,
                    "High-Vol Coverage": round(cc["high_vol_coverage"], 4),
                    "Low-Vol Coverage": round(cc["low_vol_coverage"], 4),
                    "High-Vol N": cc["high_vol_n"],
                    "Low-Vol N": cc["low_vol_n"],
                }
            )

    df_cond = pd.DataFrame(cond_rows)
    df_cond.to_csv(
        os.path.join(reports_dir, "table_4_4_conditional_coverage.csv"), index=False
    )
    print("\n── §4.4  Table 4.4: Conditional Coverage (High-Vol vs Low-Vol) ──")
    print(df_cond.to_string(index=False))
    print(
        "\n  AgACI should maintain coverage closer to target during high-vol periods."
    )

    # ──────────────────────────────────────────────────────────
    #  §4.4  FIGURES — Forecast + Rolling Coverage Plots
    # ──────────────────────────────────────────────────────────
    for m_name, m in models_data.items():
        # Figure 4.1 — Forecast vs Adaptive Intervals
        fig, ax = plt.subplots(figsize=(13, 5))
        ax.plot(y_test, label="Actual", color="black", alpha=0.7, linewidth=1.2)
        ax.plot(m["pred"], label=f"{m_name} Forecast", color="steelblue", linewidth=1.2)
        ax.fill_between(
            range(len(y_test)),
            m["l_e"], m["u_e"],
            alpha=0.2, color="steelblue", label=f"EnbPI {int((1-alpha)*100)}% PI",
        )
        ax.fill_between(
            range(len(y_test)),
            m["l_a"], m["u_a"],
            alpha=0.25, color="tomato", label=f"AgACI {int((1-alpha)*100)}% PI",
        )
        ax.set_title(f"Figure 4.1: {m_name} — Forecast vs Prediction Intervals")
        ax.set_xlabel("Test-Set Time Steps")
        ax.set_ylabel("Price (normalised)")
        ax.legend(loc="upper left")
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(reports_dir, f"{m_name}_intervals.png"), dpi=300)
        plt.close(fig)

        # Figure 4.2 — Rolling Coverage
        plot_rolling_coverage(
            y_test,
            m["l_e"], m["u_e"],
            m["l_a"], m["u_a"],
            model_name=m_name,
            window=20,
            save_path=os.path.join(reports_dir, f"{m_name}_rolling_coverage.png"),
        )

    print(f"\n✓ All Chapter 4 artefacts saved to: {reports_dir}")
    print(
        "  Next: run src/backtest.py for §5.x practical utility results."
    )


if __name__ == "__main__":
    main()

