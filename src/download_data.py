import yfinance as yf
import yaml
import os


def download_stock_data():
    # 1. Load config to get the target file path and download parameters
    config_path = os.path.join(
        os.path.dirname(__file__),
        "..",
        "configs",
        "config.yaml",
    )
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    download_config = config.get("download", {})
    ticker_symbol = download_config.get("ticker", "^GSPC")
    start_date = download_config.get("start", "2014-01-01")
    end_date = download_config.get("end", "2024-01-01")

    target_path = os.path.join(
        os.path.dirname(__file__),
        "..",
        config["data"]["file_path"],
    )

    print(
        f"Downloading data for {ticker_symbol} from {start_date} "
        f"to {end_date}..."
    )

    # 3. Fetch data
    data = yf.download(
        ticker_symbol,
        start=start_date,
        end=end_date,
        progress=False,
    )

    if data is None or data.empty:
        print(
            "Error: No data downloaded. Check your internet connection "
            "or ticker symbol."
        )
        return

    # 4. Clean formatting to match your pipeline expectations
    data = data.reset_index()

    os.makedirs(os.path.dirname(target_path), exist_ok=True)

    # 5. Save to CSV
    data.to_csv(target_path, index=False)
    print(f"Success! Data saved to: {target_path}")
    print(f"Total records: {len(data)}")


if __name__ == "__main__":
    download_stock_data()
