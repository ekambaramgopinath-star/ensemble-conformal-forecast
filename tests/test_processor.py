import numpy as np
from src.processor import DataProcessor


def test_create_sequences_and_split_data():
    data = np.arange(1, 21).reshape(-1, 1)
    dp = DataProcessor(window=5)
    X, y = dp.create_sequences(data)

    assert X.shape == (15, 5, 1)
    assert y.shape == (15, 1)
    assert np.array_equal(X[0].flatten(), np.array([1, 2, 3, 4, 5]))
    assert y[0][0] == 6

    (X_train, y_train), (X_cal, y_cal), (X_test, y_test) = dp.split_data(
        X,
        y,
        train_p=0.5,
        cal_p=0.2,
    )

    assert X_train.shape[0] == 7
    assert X_cal.shape[0] == 3
    assert X_test.shape[0] == 5
