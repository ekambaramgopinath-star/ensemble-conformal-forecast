import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


class DataProcessor:
    def __init__(self, window=30):
        self.window = window
        self.scaler = StandardScaler()

    def load_series(self, path, column='Close'):
        data = pd.read_csv(path)
        prices = pd.to_numeric(data[column], errors='coerce')
        prices = prices.dropna()
        return prices.to_numpy().reshape(-1, 1)

    def load_and_scale(self, path, column='Close'):
        prices = self.load_series(path, column)
        return self.scaler.fit_transform(prices)

    def fit_scaler(self, values):
        values = np.asarray(values).reshape(-1, 1)
        self.scaler.fit(values)

    def transform(self, values):
        values = np.asarray(values)
        shaped = values.reshape(-1, 1)
        transformed = self.scaler.transform(shaped)
        return transformed.reshape(values.shape)

    def inverse_transform(self, values):
        values = np.asarray(values)
        shaped = values.reshape(-1, 1)
        inverted = self.scaler.inverse_transform(shaped)
        return inverted.reshape(values.shape)

    def create_sequences(self, data):
        X, y = [], []
        for i in range(len(data) - self.window):
            X.append(data[i: i + self.window])
            y.append(data[i + self.window])
        return np.array(X), np.array(y)

    def split_data(self, X, y, train_p=0.6, cal_p=0.2):
        n = len(X)
        tr = int(train_p * n)
        cl = int(cal_p * n)
        return (
            (X[:tr], y[:tr]),
            (X[tr:tr+cl], y[tr:tr+cl]),
            (X[tr+cl:], y[tr+cl:])
            )
