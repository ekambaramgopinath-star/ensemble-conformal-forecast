import yfinance as yf
import yaml
import os


def download_asset(ticker_symbol, target_path, start_date, end_date):
    """Download a single asset from Yahoo Finance and save to CSV."""
    print(f"Downloading {ticker_symbol} ({start_date} → {end_date})...")

    data = yf.download(
        ticker_symbol,
        start=start_date,
        end=end_date,
        progress=False,
    )

    if data is None or data.empty:
        print(f"  ERROR: No data returned for {ticker_symbol}. Skipping.")
        return False

    data = data.reset_index()
    os.makedirs(os.path.dirname(os.path.abspath(target_path)), exist_ok=True)
    data.to_csv(target_path, index=False)
    print(f"  Saved {len(data)} rows → {target_path}")
    return True


def download_stock_data():
    config_path = os.path.join(
        os.path.dirname(__file__),
        "..",
        "configs",
        "config.yaml",
    )
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    download_config = config.get("download", {})
    start_date = download_config.get("start", "2014-01-01")
    end_date = download_config.get("end", "2024-01-01")

    # Map asset name → yfinance ticker from config
    asset_tickers = config.get("asset_tickers", {})
    asset_paths = config.get("assets", {})

    if not asset_tickers:
        # Fallback: download only the single configured asset
        ticker_symbol = download_config.get("ticker", "^GSPC")
        target_path = os.path.join(
            os.path.dirname(__file__), "..", config["data"]["file_path"]
        )
        download_asset(ticker_symbol, target_path, start_date, end_date)
        return

    # Download every asset listed in asset_tickers
    for asset_name, ticker_symbol in asset_tickers.items():
        relative_path = asset_paths.get(asset_name)
        if not relative_path:
            print(f"  WARNING: No file path configured for asset '{asset_name}'. Skipping.")
            continue

        target_path = os.path.join(
            os.path.dirname(__file__), "..", relative_path
        )
        download_asset(ticker_symbol, target_path, start_date, end_date)

    print("\nAll assets downloaded.")


if __name__ == "__main__":
    download_stock_data()
