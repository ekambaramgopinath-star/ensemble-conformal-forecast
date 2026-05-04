import os
import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# -----------------------------
# METRICS
# -----------------------------
def sharpe_ratio(returns, rf=0.0):
    returns = np.asarray(returns)
    if returns.std() == 0:
        return 0.0
    return (returns.mean() - rf) / (returns.std() + 1e-8) * np.sqrt(252)


def max_drawdown(cum_returns):
    peak = np.maximum.accumulate(cum_returns)
    drawdown = (cum_returns - peak) / (peak + 1e-8)
    return drawdown.min()


def cumulative_returns(returns):
    return np.cumprod(1 + np.asarray(returns))


# -----------------------------
# STRATEGY
# -----------------------------
def interval_trading_strategy(price, lower, upper):
    """
    Long when price < lower bound
    Short when price > upper bound
    Flat otherwise
    """

    price = np.asarray(price)
    lower = np.asarray(lower)
    upper = np.asarray(upper)

    signals = np.zeros(len(price))

    signals[price < lower] = 1     # buy
    signals[price > upper] = -1    # short

    return signals


# -----------------------------
# RETURNS SIMULATION
# -----------------------------
def simulate_pnl(price, signals, transaction_cost=0.001):
    price = np.asarray(price)
    signals = np.asarray(signals)

    returns = np.diff(price) / price[:-1]
    returns = np.append(0, returns)

    position = np.roll(signals, 1)  # enter next step
    position[0] = 0

    strategy_returns = position * returns

    # transaction costs
    trades = np.abs(np.diff(position, prepend=0))
    strategy_returns -= trades * transaction_cost

    return strategy_returns


# -----------------------------
# BACKTEST ENGINE
# -----------------------------
def run_backtest(df, model_name, method="agaci"):
    """
    df must contain:
    - Actual price
    - prediction intervals
    """

    price = df["Actual"].values

    if method == "enbpi":
        lower = df[f"{model_name}_EnbPI_Lower"].values
        upper = df[f"{model_name}_EnbPI_Upper"].values
    else:
        lower = df[f"{model_name}_AgACI_Lower"].values
        upper = df[f"{model_name}_AgACI_Upper"].values

    signals = interval_trading_strategy(price, lower, upper)
    returns = simulate_pnl(price, signals)

    equity_curve = cumulative_returns(returns)

    return returns, equity_curve, signals


# -----------------------------
# EVALUATION
# -----------------------------
def evaluate_strategy(returns, equity_curve):

    return {
        "Sharpe": sharpe_ratio(returns),
        "Max Drawdown": max_drawdown(equity_curve),
        "Total Return (%)": (equity_curve[-1] - 1) * 100,
        "Volatility": np.std(returns),
    }


# -----------------------------
# VISUALIZATION
# -----------------------------
def plot_equity_curves(results, save_path):

    plt.figure(figsize=(12, 6))

    for label, eq in results.items():
        plt.plot(eq, label=label)

    plt.title("Strategy Equity Curve Comparison")
    plt.xlabel("Time")
    plt.ylabel("Cumulative Returns")
    plt.legend()
    plt.grid(alpha=0.3)

    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


# -----------------------------
# MAIN
# -----------------------------
def main():

    config_path = os.path.join(
        os.path.dirname(__file__),
        "..",
        "configs",
        "config.yaml",
    )

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    results_dir = os.path.join(os.path.dirname(__file__), "..", "results")
    reports_dir = os.path.join(os.path.dirname(__file__), "..", "reports")
    os.makedirs(reports_dir, exist_ok=True)

    # -----------------------------
    # LOAD LATEST RESULTS
    # -----------------------------
    files = [
        os.path.join(results_dir, f)
        for f in os.listdir(results_dir)
        if f.endswith(".csv")
    ]

    if not files:
        print("No results found. Run pipeline first.")
        return

    latest_file = max(files, key=os.path.getctime)
    print(f"Using: {latest_file}")

    df = pd.read_csv(latest_file)

    models = ["LSTM", "GRU"]
    methods = ["enbpi", "agaci"]

    summary = []
    equity_curves = {}

    # -----------------------------
    # BACKTEST ALL MODELS
    # -----------------------------
    for model in models:
        for method in methods:

            returns, equity, signals = run_backtest(df, model, method)

            metrics = evaluate_strategy(returns, equity)

            summary.append({
                "Model": model,
                "Method": method.upper(),
                **metrics,
            })

            equity_curves[f"{model}-{method}"] = equity

    # -----------------------------
    # SAVE RESULTS
    # -----------------------------
    summary_df = pd.DataFrame(summary)

    summary_path = os.path.join(reports_dir, "backtest_summary.csv")
    summary_df.to_csv(summary_path, index=False)

    print("\n=== BACKTEST RESULTS ===")
    print(summary_df)

    # -----------------------------
    # PLOT EQUITY CURVES
    # -----------------------------
    plot_path = os.path.join(reports_dir, "equity_curves.png")
    plot_equity_curves(equity_curves, plot_path)

    print(f"\nSaved: {summary_path}")
    print(f"Saved: {plot_path}")


# -----------------------------
if __name__ == "__main__":
    main()