import numpy as np
import matplotlib.pyplot as plt

# 1. Create dummy data that looks like a stock price
np.random.seed(42)
time = np.arange(100)
# Synthetic price with an upward trend and some noise
price = 100 + 0.5 * time + np.cumsum(np.random.normal(0, 1, 100))

# 2. Define the "Forecast" area (last 20 points)
forecast_start = 80
history = price[:forecast_start]
actual_future = price[forecast_start:]

# 3. Create a "Point Forecast" (a simple linear trend line)
point_forecast = history[-1] + 0.4 * np.arange(1, 21)

# 4. Create "Intervals" (widening uncertainty over time)
# In reality, this would come from your EnbPI or AgACI code
uncertainty = np.linspace(1, 8, 20)
upper_bound = point_forecast + uncertainty
lower_bound = point_forecast - uncertainty

# 5. Plotting
plt.figure(figsize=(12, 6))

# Plot historical data
plt.plot(
    time[:forecast_start],
    history,
    color='black',
    label='Historical Price',
    linewidth=1.5,
)

# Plot actual future (for comparison)
plt.plot(
    time[forecast_start:],
    actual_future,
    color='black',
    linestyle='--',
    alpha=0.5,
    label='Actual Realization',
)

# Plot Point Forecast (The "Line")
plt.plot(
    time[forecast_start:],
    point_forecast,
    color='red',
    marker='o',
    markersize=4,
    label='Point Forecast (Standard DL)',
)

# Plot Interval Forecast (The "Shaded Area")
plt.fill_between(
    time[forecast_start:],
    lower_bound,
    upper_bound,
    color='red',
    alpha=0.2,
    label='Prediction Interval (90% Confidence)',
)

# Formatting
plt.title(
    'Deterministic Point Forecast vs. '
    'Probabilistic Interval Forecast',
    fontsize=14,
    fontweight='bold',
)
plt.xlabel('Time Steps', fontsize=12)
plt.ylabel('Asset Price', fontsize=12)
plt.legend(loc='upper left')
plt.grid(True, linestyle=':', alpha=0.6)

# Save the figure
plt.savefig('point_vs_interval_comparison.png', dpi=300)
plt.show()
