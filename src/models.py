import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, GRU, Dense, Dropout


def build_model(
    model_type,
    input_shape,
    hidden_units=(64, 32),
    dropout_rate=0.2,
):
    model = Sequential()
    if len(hidden_units) != 2:
        raise ValueError(
            "hidden_units must contain two values: [first_layer, second_layer]"
        )

    first_units, second_units = hidden_units
    if model_type == "LSTM":
        model.add(
            LSTM(first_units, return_sequences=True, input_shape=input_shape)
        )
        model.add(Dropout(dropout_rate))
        model.add(LSTM(second_units))
    elif model_type == "GRU":
        model.add(
            GRU(first_units, return_sequences=True, input_shape=input_shape)
        )
        model.add(Dropout(dropout_rate))
        model.add(GRU(second_units))
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    model.add(Dense(1))
    model.compile(optimizer="adam", loss="mse")
    return model


class EnsembleWrapper:
    def __init__(
        self,
        model_type,
        B=5,
        hidden_units=(64, 32),
        dropout_rate=0.2,
        batch_size=32,
    ):
        self.model_type = model_type
        self.B = B
        self.hidden_units = hidden_units
        self.dropout_rate = dropout_rate
        self.batch_size = batch_size
        self.models = []

    def fit(self, X, y, epochs=15):
        self.models = []
        for i in range(self.B):
            print(f"  Training {self.model_type} member {i+1}/{self.B}...")
            m = build_model(
                self.model_type,
                (X.shape[1], X.shape[2]),
                hidden_units=self.hidden_units,
                dropout_rate=self.dropout_rate,
            )
            m.fit(
                X,
                y,
                epochs=epochs,
                batch_size=self.batch_size,
                verbose=0,
            )
            self.models.append(m)

    def predict(self, X):
        preds = [m.predict(X, verbose=0).flatten() for m in self.models]
        return np.mean(preds, axis=0)
