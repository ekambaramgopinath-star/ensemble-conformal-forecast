import os
import yaml
import numpy as np
import pandas as pd

from src.processor import DataProcessor
from src.models import EnsembleWrapper
from src.conformal import get_residuals, apply_enbpi, rolling_agaci
from src.evaluation import rmse, mape, coverage, interval_width


def run_asset(name, path, config):

    print(f"\nRunning asset: {name}")

    dp = DataProcessor(window=config["data"]["window_size"])

    prices = dp.load_series(path, column=config["data"]["target_column"])
    returns = dp.to_returns(prices)
    X, y = dp.create_sequences(returns.reshape(-1, 1))

    (X_tr, y_tr), (X_cal, y_cal), (X_te, y_te) = dp.split_data(
        X, y,
        config["data"]["train_split"],
        config["data"]["cal_split"]
    )

    dp.fit_scaler(X_tr)

    X_tr = dp.transform(X_tr)
    X_cal = dp.transform(X_cal)
    X_te = dp.transform(X_te)

    y_tr = dp.transform(y_tr)
    y_cal = dp.transform(y_cal)
    y_te = dp.transform(y_te)

    model = EnsembleWrapper("LSTM", B=config["model"]["ensemble_size"])
    model.fit(X_tr, y_tr, epochs=config["model"]["epochs"])

    cal_pred = model.predict(X_cal)
    test_pred = model.predict(X_te)

    residuals = get_residuals(y_cal, cal_pred)

    l_e, u_e = apply_enbpi(test_pred, residuals, config["conformal"]["alpha"])
    l_a, u_a = rolling_agaci(
        test_pred, y_te, residuals,
        alpha=config["conformal"]["alpha"],
        window=config["conformal"]["agaci_window"]
    )

    y_te = dp.inverse(y_te)
    test_pred = dp.inverse(test_pred)
    l_e = dp.inverse(l_e)
    u_e = dp.inverse(u_e)
    l_a = dp.inverse(l_a)
    u_a = dp.inverse(u_a)

    return {
        "RMSE": rmse(y_te, test_pred),
        "MAPE": mape(y_te, test_pred),
        "EnbPI_Coverage": coverage(y_te, l_e, u_e),
        "AgACI_Coverage": coverage(y_te, l_a, u_a),
        "EnbPI_Width": interval_width(l_e, u_e),
        "AgACI_Width": interval_width(l_a, u_a),
    }


def main():

    config = yaml.safe_load(
        open(
            os.path.join(
                os.path.dirname(__file__), "..", "configs", "config.yaml"
            )
        )
    )

    results = []

    for asset, path in config["assets"].items():
        asset_path = os.path.join(
            os.path.dirname(__file__), "..", path
        )
        metrics = run_asset(asset, asset_path, config)
        metrics["Asset"] = asset
        results.append(metrics)

    df = pd.DataFrame(results)

    reports_dir = os.path.join(
        os.path.dirname(__file__), "..", "reports", "asset_comparison"
    )
    os.makedirs(reports_dir, exist_ok=True)
    df.to_csv(os.path.join(reports_dir, "summary.csv"), index=False)

    print("\nFINAL RESULTS:")
    print(df)


if __name__ == "__main__":
    main()