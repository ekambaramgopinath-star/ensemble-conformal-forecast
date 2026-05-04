import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# -----------------------------
# CONFIG
# -----------------------------
BASE_PATH = Path(".")
REPORT_PATH = BASE_PATH / "reports"
REPORT_PATH.mkdir(exist_ok=True)

sns.set(style="whitegrid", context="paper", font_scale=1.2)

# -----------------------------
# LOAD DATA
# -----------------------------
point_df = pd.read_csv(REPORT_PATH / "table_4_2_point_accuracy.csv")
interval_df = pd.read_csv(REPORT_PATH / "table_4_3_interval_metrics.csv")
cond_df = pd.read_csv(REPORT_PATH / "table_4_4_conditional_coverage.csv")

# -----------------------------
# 1. TABLE EXPORT (LaTeX + CSV)
# -----------------------------


def save_latex_table(df, name):
    latex_str = df.to_latex(index=False, float_format="%.4f")
    with open(REPORT_PATH / f"{name}.tex", "w") as f:
        f.write(latex_str)


save_latex_table(point_df, "point_accuracy")
save_latex_table(interval_df, "interval_metrics")
save_latex_table(cond_df, "conditional_coverage")

# -----------------------------
# 2. POINT FORECAST BAR PLOTS
# -----------------------------
plt.figure(figsize=(6, 4))
sns.barplot(data=point_df, x="Model", y="RMSE", palette="muted")
plt.title("RMSE Comparison Across Models")
plt.tight_layout()
plt.savefig(REPORT_PATH / "rmse_comparison.png", dpi=300)
plt.close()

plt.figure(figsize=(6, 4))
sns.barplot(data=point_df, x="Model", y="MAPE (%)", palette="muted")
plt.title("MAPE Comparison Across Models")
plt.tight_layout()
plt.savefig(REPORT_PATH / "mape_comparison.png", dpi=300)
plt.close()

# -----------------------------
# 3. COVERAGE VS WIDTH TRADEOFF
# -----------------------------
plt.figure(figsize=(6, 5))
sns.scatterplot(
    data=interval_df,
    x="Avg Width",
    y="Empirical Coverage",
    hue="Method",
    style="Model",
    s=100
)
plt.axhline(0.9, linestyle="--", color="red", label="Target 90%")
plt.title("Coverage vs Interval Width Trade-off")
plt.legend()
plt.tight_layout()
plt.savefig(REPORT_PATH / "coverage_vs_width.png", dpi=300)
plt.close()

# -----------------------------
# 4. CONDITIONAL COVERAGE PLOT
# -----------------------------
cond_melt = cond_df.melt(
    id_vars=["Model", "Method"],
    value_vars=["High-Vol Coverage", "Low-Vol Coverage"],
    var_name="Regime",
    value_name="Coverage"
)

plt.figure(figsize=(7, 5))
sns.barplot(
    data=cond_melt,
    x="Model",
    y="Coverage",
    hue="Regime"
)
plt.axhline(0.9, linestyle="--", color="black")
plt.title("Coverage Under Different Volatility Regimes")
plt.tight_layout()
plt.savefig(REPORT_PATH / "conditional_coverage.png", dpi=300)
plt.close()

# -----------------------------
# 5. INTERVAL WIDTH COMPARISON
# -----------------------------
plt.figure(figsize=(6, 4))
sns.barplot(
    data=interval_df,
    x="Model",
    y="Avg Width",
    hue="Method"
)
plt.title("Prediction Interval Width Comparison")
plt.tight_layout()
plt.savefig(REPORT_PATH / "interval_width.png", dpi=300)
plt.close()

# -----------------------------
# 6. ROLLING COVERAGE (FROM PREDICTIONS)
# -----------------------------


def plot_rolling_coverage(file_path, name):
    df = pd.read_csv(file_path)

    # Expected columns:
    # y_true, y_lower, y_upper
    df["covered"] = (
        (df["y_true"] >= df["y_lower"])
        &
        (df["y_true"] <= df["y_upper"])
    ).astype(int)

    df["rolling_cov"] = df["covered"].rolling(window=50).mean()

    plt.figure(figsize=(8, 4))
    plt.plot(df["rolling_cov"], label="Rolling Coverage")
    plt.axhline(0.9, linestyle="--", color="red", label="Target")
    plt.title(f"Rolling Coverage ({name})")
    plt.legend()
    plt.tight_layout()
    plt.savefig(REPORT_PATH / f"rolling_coverage_{name}.png", dpi=300)
    plt.close()


# Apply to all prediction files
for f in (BASE_PATH / "results").glob("test_predictions_*.csv"):
    plot_rolling_coverage(f, f.stem)

# -----------------------------
# DONE
# -----------------------------
print("All tables and plots generated in /reports")
