import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


class DataProcessor:
    def __init__(self, window=30):
        self.window = window
        self.scaler = StandardScaler()

    # ----------------------------
    # 1. LOAD PRICE SERIES
    # ----------------------------
    def load_series(self, path, column="Close"):
        df = pd.read_csv(path)
        prices = pd.to_numeric(df[column], errors="coerce").dropna()
        return prices.values.astype(float)

    # ----------------------------
    # 2. CONVERT TO LOG RETURNS
    # ----------------------------
    def to_returns(self, prices):
        returns = np.log(prices[1:] / prices[:-1])
        return returns

    # ----------------------------
    # 3. CREATE SEQUENCES (ON RETURNS)
    # ----------------------------
    def create_sequences(self, data):
        X, y = [], []
        for i in range(len(data) - self.window):
            X.append(data[i:i+self.window])
            y.append(data[i+self.window])
        return np.array(X), np.array(y)

    # ----------------------------
    # 4. TRAIN / CAL / TEST SPLIT
    # ----------------------------
    def split_data(self, X, y, train_p=0.6, cal_p=0.2):
        n = len(X)
        tr = int(train_p * n)
        cl = int(cal_p * n)

        return (
            (X[:tr], y[:tr]),
            (X[tr:tr+cl], y[tr:tr+cl]),
            (X[tr+cl:], y[tr+cl:])
        )

    # ----------------------------
    # 5. SCALING (ON RETURNS)
    # ----------------------------
    def fit_scaler(self, train_data):
        self.scaler.fit(train_data.reshape(-1, 1))

    def transform(self, data):
        shape = data.shape
        return self.scaler.transform(data.reshape(-1, 1)).reshape(shape)

    def inverse(self, data):
        return self.scaler.inverse_transform(
            np.asarray(data).reshape(-1, 1)
        ).flatten()

    def inverse_transform(self, data):
        return self.inverse(data)

    # ----------------------------
    # 6. RECONSTRUCT PRICES FROM RETURNS
    # ----------------------------
    def reconstruct_prices(self, last_price, predicted_returns):
        prices = [last_price]
        for r in predicted_returns:
            prices.append(prices[-1] * np.exp(r))
        return np.array(prices[1:])