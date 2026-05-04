import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# --- 1. Load your provided data ---
# Make sure these filenames match your local files exactly.
try:
    # Use Path to construct paths relative to this script's location
    script_dir = Path(__file__).parent
    data_dir = script_dir.parent / 'data'
    sp500_data = pd.read_csv(data_dir / 'sp500.csv').iloc[1:].reset_index(
        drop=True)
    btc_data = pd.read_csv(data_dir / 'btc.csv').iloc[1:].reset_index(
        drop=True)
    fx_data = pd.read_csv(data_dir / 'fx.csv').iloc[1:].reset_index(
        drop=True)
except FileNotFoundError as e:
    msg = (
        f"Error: Make sure your CSV files are in the same folder. "
        f"Detailed error: {e}"
    )
    print(msg)
    # Make sure the paths are correct in your final execution
    # environment.
    exit()


# --- 2. Data Cleaning and Log Return Calculation ---
def preprocess_and_calc_returns(df, price_col_name):
    """
    Standardizes dates, calculates log returns, and sets date as index.
    Log Returns = ln(Pt / Pt-1).
    """
    # 2a. Convert Date column to datetime objects robustly.
    df['Date'] = pd.to_datetime(df['Date'])

    # 2b. Convert price column to numeric.
    df[price_col_name] = pd.to_numeric(
        df[price_col_name], errors='coerce'
    )

    # 2c. Sort to ensure temporal order.
    df = df.sort_values(by='Date')

    # 2d. Calculate Log Returns. standard in econometrics.
    # ln(Pt) - ln(Pt-1) = ln(Pt/Pt-1).
    # .shift(1) aligns the denominator Pt-1 with Pt.
    log_returns = np.log(df[price_col_name]) - np.log(
        df[price_col_name].shift(1)
    )
    df['Log Returns'] = pd.Series(log_returns)

    # 2e. Set Date as Index for clean plotting.
    df = df.set_index('Date')

    # 2f. Handle potential NaN (first row after shift).
    df = df.dropna()

    return df


# Apply the preprocessing. Guessing common column names based on
# standard file sources.
# !!! IMPORTANT !!! Update 'price_col_name' to match your exact CSV
# headers if needed.
processed_sp500 = preprocess_and_calc_returns(
    sp500_data, price_col_name='Close'
)
processed_btc = preprocess_and_calc_returns(
    btc_data, price_col_name='Close'
)
processed_fx = preprocess_and_calc_returns(
    fx_data, price_col_name='Close'
)

# --- 3. Create the Multi-Pane Plot [Figure 1.1] ---
# Create figure with 3 stacked subplots sharing the X-axis (Dates).
fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)


# Define a function to style the return plots consistently.
def plot_returns(ax, df, title_label, color_hex):
    """Plots log returns on a specific axis with consistent formatting."""
    ax.plot(
        df.index,
        df['Log Returns'],
        label=f'{title_label} Log Returns',
        color=color_hex,
        linewidth=1,
    )
    ax.set_title(
        f'Historical Daily Log Returns - {title_label}',
        fontsize=14,
        fontweight='bold',
    )
    ax.set_ylabel('Log Returns', fontsize=12)
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    ax.legend(loc='upper right')


# Plot each asset class.
plot_returns(
    axes[0], processed_sp500, 'Equities (S&P 500)', '#3498db'
)  # Blue
plot_returns(
    axes[1], processed_btc, 'Cryptocurrencies (Bitcoin)', '#f39c12'
)  # Orange
plot_returns(
    axes[2], processed_fx, 'Foreign Exchange (EUR/USD)', '#2ecc71'
)  # Green

# Format the final shared X-axis.
axes[2].set_xlabel('Date', fontsize=12)
plt.xticks(rotation=45)

# Adjust layout to prevent overlap and save.
plt.tight_layout()

# Save the figure so you can insert it into your thesis.
output_plot_path = 'final_figure_1_1.png'
plt.savefig(output_plot_path)

print(f"Figure 1.1 has been generated and saved as: {output_plot_path}")
