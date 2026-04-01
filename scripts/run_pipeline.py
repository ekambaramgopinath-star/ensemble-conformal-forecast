import os
import random
import shutil
import yaml
import numpy as np
import pandas as pd
import tensorflow as tf
from datetime import datetime

from src.processor import DataProcessor
from src.models import EnsembleWrapper
from src.conformal import get_residuals, apply_enbpi, rolling_agaci
from src.visualization import calculate_metrics


def main():
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] Starting Thesis Pipeline...")

    config_path = os.path.join(
        os.path.dirname(__file__),
        "..",
        "configs",
        "config.yaml",
    )
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    seed = config.get("seed", 42)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

    print("Step 1: Loading and preprocessing data...")
    dp = DataProcessor(window=config["data"]["window_size"])

    data_path = os.path.join(
        os.path.dirname(__file__),
        "..",
        config["data"]["file_path"],
    )
    raw_data = dp.load_series(
        data_path,
        column=config["data"]["target_column"],
    )
    X, y = dp.create_sequences(raw_data)

    (X_train, y_train), (X_cal, y_cal), (X_test, y_test) = dp.split_data(
        X,
        y,
        train_p=config["data"]["train_split"],
        cal_p=config["data"]["cal_split"],
    )

    dp.fit_scaler(np.vstack((X_train.reshape(-1, 1), y_train.reshape(-1, 1))))
    X_train = dp.transform(X_train)
    y_train = dp.transform(y_train)
    X_cal = dp.transform(X_cal)
    y_cal = dp.transform(y_cal)
    X_test = dp.transform(X_test)
    y_test = dp.transform(y_test)

    print(
        f"Step 2: Training LSTM Ensemble "
        f"(B={config['model']['ensemble_size']})..."
    )
    lstm_ens = EnsembleWrapper(
        "LSTM",
        B=config["model"]["ensemble_size"],
        hidden_units=tuple(config["model"]["hidden_units"]),
        dropout_rate=config["model"]["dropout_rate"],
        batch_size=config["model"]["batch_size"],
    )
    lstm_ens.fit(X_train, y_train, epochs=config["model"]["epochs"])

    print(
        f"Step 3: Training GRU Ensemble "
        f"(B={config['model']['ensemble_size']})..."
    )
    gru_ens = EnsembleWrapper(
        "GRU",
        B=config["model"]["ensemble_size"],
        hidden_units=tuple(config["model"]["hidden_units"]),
        dropout_rate=config["model"]["dropout_rate"],
        batch_size=config["model"]["batch_size"],
    )
    gru_ens.fit(X_train, y_train, epochs=config["model"]["epochs"])

    print("Step 4: Running conformal inference (EnbPI & AgACI)...")

    lstm_cal_pred = lstm_ens.predict(X_cal)
    gru_cal_pred = gru_ens.predict(X_cal)

    lstm_res = get_residuals(y_cal, lstm_cal_pred)
    gru_res = get_residuals(y_cal, gru_cal_pred)

    lstm_test_pred = lstm_ens.predict(X_test)
    gru_test_pred = gru_ens.predict(X_test)
    baseline_pred = X_test[:, -1, 0].flatten()

    l_enbpi, u_enbpi = apply_enbpi(
        lstm_test_pred,
        lstm_res,
        alpha=config["conformal"]["alpha"],
    )
    g_enbpi, gu_enbpi = apply_enbpi(
        gru_test_pred,
        gru_res,
        alpha=config["conformal"]["alpha"],
    )

    l_agaci, u_agaci = rolling_agaci(
        lstm_test_pred,
        y_test,
        lstm_res,
        alpha=config["conformal"]["alpha"],
        window=config["conformal"]["agaci_window"],
    )
    g_agaci, gu_agaci = rolling_agaci(
        gru_test_pred,
        y_test,
        gru_res,
        alpha=config["conformal"]["alpha"],
        window=config["conformal"]["agaci_window"],
    )

    print("Step 5: Saving results...")

    def inv(values):
        values = np.asarray(values).reshape(-1, 1)
        return dp.inverse_transform(values).flatten()

    y_test_raw = inv(y_test)
    baseline_raw = inv(baseline_pred)
    lstm_test_pred_raw = inv(lstm_test_pred)
    gru_test_pred_raw = inv(gru_test_pred)
    l_enbpi_raw = inv(l_enbpi)
    u_enbpi_raw = inv(u_enbpi)
    g_enbpi_raw = inv(g_enbpi)
    gu_enbpi_raw = inv(gu_enbpi)
    l_agaci_raw = inv(l_agaci)
    u_agaci_raw = inv(u_agaci)
    g_agaci_raw = inv(g_agaci)
    gu_agaci_raw = inv(gu_agaci)

    lstm_enbpi_cov, lstm_enbpi_width = calculate_metrics(
        y_test_raw,
        l_enbpi_raw,
        u_enbpi_raw,
    )
    gru_enbpi_cov, gru_enbpi_width = calculate_metrics(
        y_test_raw,
        g_enbpi_raw,
        gu_enbpi_raw,
    )
    lstm_agaci_cov, lstm_agaci_width = calculate_metrics(
        y_test_raw,
        l_agaci_raw,
        u_agaci_raw,
    )
    gru_agaci_cov, gru_agaci_width = calculate_metrics(
        y_test_raw,
        g_agaci_raw,
        gu_agaci_raw,
    )

    baseline_rmse = np.sqrt(np.mean((y_test_raw - baseline_raw) ** 2))
    denominator = np.where(np.abs(y_test_raw) < 1e-8, 1e-8, y_test_raw)
    baseline_mape = np.mean(
        np.abs((y_test_raw - baseline_raw) / denominator)
    ) * 100

    results_df = pd.DataFrame(
        {
            "Actual": y_test_raw,
            "Baseline_Pred": baseline_raw,
            "LSTM_Pred": lstm_test_pred_raw,
            "GRU_Pred": gru_test_pred_raw,
            "LSTM_EnbPI_Lower": l_enbpi_raw,
            "LSTM_EnbPI_Upper": u_enbpi_raw,
            "LSTM_AgACI_Lower": l_agaci_raw,
            "LSTM_AgACI_Upper": u_agaci_raw,
            "GRU_EnbPI_Lower": g_enbpi_raw,
            "GRU_EnbPI_Upper": gu_enbpi_raw,
            "GRU_AgACI_Lower": g_agaci_raw,
            "GRU_AgACI_Upper": gu_agaci_raw,
        }
    )

    results_dir = os.path.join(
        os.path.dirname(__file__),
        "..",
        "results",
    )
    os.makedirs(results_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    config_copy_path = os.path.join(results_dir, f"config_{timestamp}.yaml")
    shutil.copyfile(config_path, config_copy_path)

    results_df.to_csv(
        f"{results_dir}/test_predictions_{timestamp}.csv",
        index=False,
    )

    print("\n" + "=" * 30)
    print("PIPELINE SUMMARY")
    print(f"Baseline RMSE: {baseline_rmse:.4f}")
    print(f"Baseline MAPE: {baseline_mape:.2f}%")
    print(f"LSTM EnbPI Coverage: {lstm_enbpi_cov:.2%}")
    print(f"LSTM AgACI Coverage: {lstm_agaci_cov:.2%}")
    print(f"GRU EnbPI Coverage: {gru_enbpi_cov:.2%}")
    print(f"GRU AgACI Coverage: {gru_agaci_cov:.2%}")
    print(
        f"Results saved to: results/test_predictions_{timestamp}.csv"
    )
    print("=" * 30)


if __name__ == "__main__":
    main()
