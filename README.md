# 📁 ensemble-conformal-forecast: Adaptive Conformal Inference for Financial Time-Series

A clean and modular prototype for forecasting financial indices (such as the S&P 500) with uncertainty quantification. The project compares:

* **Ensemble Batch Prediction Intervals (EnbPI)**
* **Adaptive Group-Conditional Conformal Inference (AgACI)**

---

## 🚀 Key Features

* **Hybrid Ensemble Architecture** — LSTM and GRU ensembles capture temporal patterns.
* **Quantified Uncertainty** — Predictive intervals instead of only point forecasts.
* **Adaptive Calibration** — AgACI updates its interval width dynamically during volatility.
* **Baseline Comparison** — includes a persistence baseline for better performance context.
* **Modular Codebase** — Data, modeling, conformal logic, and visualization are separated cleanly.

---

## 📂 Repository Structure

```text
ensemble-conformal-forecast/
├── configs/
│   └── config.yaml           # Hyperparameters and file paths
├── data/
│   └── sp500.csv             # Downloaded historical price data
├── notebooks/
│   ├── 01_exploration.ipynb  # EDA and result exploration
│   └── sensitivity_analysis.ipynb
├── results/
│   └── ...                   # Generated predictions and outputs
├── reports/
│   └── ...                   # Plots and CSV tables for Chapter 4
├── scripts/
│   ├── run_pipeline.py       # Main pipeline entrypoint (SP500)
│   ├── run_assets.py         # Multi-asset comparison pipeline
│   └── generate_report.py    # Bar-chart report generator
├── src/
│   ├── backtest.py           # Trading strategy simulation & equity curves
│   ├── conformal.py          # EnbPI and AgACI logic
│   ├── download_data.py      # Data download utilities (yfinance)
│   ├── evaluation.py         # Shared metric functions
│   ├── generate_results.py   # Result tables and plots (Chapter 4)
│   ├── models.py             # LSTM/GRU ensemble definitions
│   ├── processor.py          # Scaling and sequence generation
│   ├── stats_test.py         # Cross-asset statistical comparison helpers
│   ├── tuning.py             # Optuna hyperparameter tuning stub
│   └── visualization.py      # Plotting helpers
├── tests/
│   ├── test_conformal.py     # Unit tests for conformal logic
│   └── test_processor.py     # Unit tests for data processing
├── Dockerfile                # Container build spec
├── docker-compose.yml        # Full workflow container orchestration
├── README.md                 # This document
└── requirements.txt          # Python dependencies
```

---

## 🛠 Installation & Usage

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure the experiment

Edit `configs/config.yaml` and set:

* `window_size` — look-back length for sequence creation.
* `alpha` — conformal miscoverage rate (e.g. `0.1` for 90% intervals).
* `ensemble_size` — number of ensemble members.

### 3. Run the pipeline

Run these in order:

```bash
python src/download_data.py
python scripts/run_pipeline.py
python src/generate_results.py
```

Everything is now verified and ready for your thesis defense!

Outputs are saved automatically into `results/`.

### 3.1 Run the pipeline with Docker

If your PC cannot install Keras or TensorFlow directly, use Docker instead.

Build the image:

```bash
docker build -t ensemble-conformal-forecast .
```

Run the full workflow (Linux/macOS or cmd.exe on Windows):

```bash
docker run --rm -v "%cd%/data:/app/data" -v "%cd%/results:/app/results" -v "%cd%/reports:/app/reports" ensemble-conformal-forecast sh -c "python src/download_data.py && python scripts/run_pipeline.py && python src/generate_results.py"
```

Run the full workflow in PowerShell:

```powershell
docker run --rm `
  -v "${PWD}:/app" `
  -v "${PWD}/data:/app/data" `
  -v "${PWD}/results:/app/results" `
  -v "${PWD}/reports:/app/reports" `
  -e PYTHONPATH=/app `
  ensemble-conformal-forecast `
  sh -c "python src/download_data.py; python scripts/run_pipeline.py; python src/generate_results.py"
```

```powershell
docker run --rm `
  -v "${PWD}:/app" `
  -v "${PWD}/data:/app/data" `
  -v "${PWD}/results:/app/results" `
  -v "${PWD}/reports:/app/reports" `
   -v "${PWD}/reports:/app/reports/asset_comparison" `
  -e PYTHONPATH=/app `
  ensemble-conformal-forecast `
  sh -c "python src/backtest.py; python scripts/run_assets.py; python scripts/generate_report.py"
```

```powershell
docker run --rm `
  -v "${PWD}:/app" `
  -w /app `
  -e PYTHONPATH=/app `
  ensemble-conformal-forecast `
  sh -c "jupyter nbconvert --to notebook --execute notebooks/sensitivity_analysis.ipynb --output executed/sa_result.ipynb --ExecutePreprocessor.timeout=600"
```

```powershell
docker run --rm `
  -v "${PWD}:/app" `
  -w /app `
  -e PYTHONPATH=/app `
  ensemble-conformal-forecast `
  sh -c "jupyter nbconvert --to notebook --execute notebooks/01_exploration.ipynb --output executed/full_result.ipynb --ExecutePreprocessor.timeout=600"
```

Or use Docker Compose:

```bash
docker compose up --build
```

If your Docker version still uses the legacy compose CLI, run:

```bash
docker-compose up --build
```

### 4. Review results

code is located in `notebooks/01_exploration.ipynb`

Open `notebooks/executed/executed_notebook.ipynb` (or `result1.ipynb` / `result2.ipynb`) to inspect:

* point forecast performance,
* interval coverage,
* comparison between EnbPI and AgACI.

---

## 🧠 Methodology

### EnbPI

Provides global prediction intervals using ensemble residuals from a calibration set. It aims to guarantee average coverage across the test set.

### AgACI

Uses a rolling window of recent residuals to adapt prediction intervals during regime changes. This improves local coverage in volatile periods.

---

## 📊 Evaluation Framework (per Thesis Chapter)

| Thesis Section | Metric / Test | Code Location |
|---|---|---|
| §4.2 Point Forecast Accuracy | RMSE, MAE, MAPE | `src/evaluation.py` → `rmse`, `mae`, `mape` |
| §4.2 Statistical Comparison | Diebold-Mariano test (LSTM vs GRU) | `src/stats_test.py` → `diebold_mariano` |
| §4.3 Interval Quality | Empirical Coverage, Coverage Deviation | `src/evaluation.py` → `coverage`, `coverage_deviation` |
| §4.3 Interval Efficiency | Average Interval Width | `src/evaluation.py` → `interval_width` |
| §4.3 Proper Scoring Rule | Winkler Score (penalises width + violations) | `src/evaluation.py` → `winkler_score` |
| §4.3 Formal Calibration Test | Kupiec POF test (chi-squared, H₀: failure rate = α) | `src/evaluation.py` → `kupiec_pof_test` |
| §4.4 Adaptive Behaviour | 20-day rolling coverage plot | `src/visualization.py` → `plot_rolling_coverage` |
| §4.4 Regime-Conditional Coverage | High-vol vs low-vol coverage split | `src/evaluation.py` → `conditional_coverage` |
| §4.5 Sensitivity Analysis | α, window, ensemble size sweeps | `notebooks/sensitivity_analysis.ipynb` |
| §5.x Practical Utility | Sharpe Ratio, Max Drawdown, Total Return | `src/backtest.py` → `evaluate_strategy` |

---

## 📊 Evaluation Metrics

* **RMSE** — Root mean Squared Error for point forecast quality.
* **Coverage** — Fraction of true values inside the interval (target = `1 - alpha`).
* **Interval Width** — Average prediction interval size.

---

## ✅ Notes

* Make sure `configs/config.yaml` points to the correct dataset path.
* Run `src/download_data.py` first to populate `data/sp500.csv`.
* Run `scripts/run_pipeline.py` before `src/generate_results.py`.
* Each pipeline run saves a copy of the active config in `results/` for reproducibility.
* Validate core logic with `python -m unittest discover -s tests`.

---

check the full results under results and reports folders

---

**Author:** [Gopinath Ekambaram]  
**Academic Year:** 2026
