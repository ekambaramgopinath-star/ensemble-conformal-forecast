import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, GRU, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping


# ============================================================
# MODEL BUILDER
# ============================================================
def build_model(
    model_type,
    input_shape,
    hidden_units=(64, 32),
    dropout_rate=0.2,
    learning_rate=0.001,
):
    model = Sequential()

    for i, units in enumerate(hidden_units):
        return_seq = i < len(hidden_units) - 1

        if model_type == "LSTM":
            layer = LSTM(
                units,
                return_sequences=return_seq,
                input_shape=input_shape if i == 0 else None,
            )
        elif model_type == "GRU":
            layer = GRU(
                units,
                return_sequences=return_seq,
                input_shape=input_shape if i == 0 else None,
            )
        else:
            raise ValueError(f"Unknown model_type: {model_type}")

        model.add(layer)
        model.add(Dropout(dropout_rate))

    model.add(Dense(1))

    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss="mse")

    return model


# ============================================================
# ENSEMBLE WRAPPER
# ============================================================
class EnsembleWrapper:
    def __init__(
        self,
        model_type,
        B=5,
        hidden_units=(64, 32),
        dropout_rate=0.2,
        batch_size=32,
        learning_rate=0.001,
        bootstrap=True,
    ):
        self.model_type = model_type
        self.B = B
        self.hidden_units = hidden_units
        self.dropout_rate = dropout_rate
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.bootstrap = bootstrap

        self.models = []

    # -----------------------------
    # BOOTSTRAP SAMPLING
    # -----------------------------
    def _resample(self, X, y):
        if not self.bootstrap:
            return X, y

        idx = np.random.choice(len(X), size=len(X), replace=True)
        return X[idx], y[idx]

    # -----------------------------
    # TRAIN
    # -----------------------------
    def fit(self, X, y, epochs=15):
        self.models = []

        for i in range(self.B):
            print(f"Training {self.model_type} member {i+1}/{self.B}")

            X_res, y_res = self._resample(X, y)

            model = build_model(
                self.model_type,
                (X.shape[1], X.shape[2]),
                hidden_units=self.hidden_units,
                dropout_rate=self.dropout_rate,
                learning_rate=self.learning_rate,
            )

            early_stop = EarlyStopping(
                monitor="loss",
                patience=3,
                restore_best_weights=True,
            )

            model.fit(
                X_res,
                y_res,
                epochs=epochs,
                batch_size=self.batch_size,
                verbose=0,
                callbacks=[early_stop],
            )

            self.models.append(model)

    # ============================================================
    # PREDICTION (MEAN ENSEMBLE)
    # ============================================================
    def predict(self, X):
        preds = np.array(
            [m.predict(X, verbose=0).flatten() for m in self.models]
        )
        return np.mean(preds, axis=0)

    # ============================================================
    # PREDICTION + UNCERTAINTY (IMPORTANT UPGRADE)
    # ============================================================
    def predict_with_uncertainty(self, X):
        preds = np.array(
            [m.predict(X, verbose=0).flatten() for m in self.models]
        )

        mean = np.mean(preds, axis=0)
        std = np.std(preds, axis=0)

        return mean, std

    # ============================================================
    # MONTE CARLO DROPOUT INFERENCE (NEW - IMPORTANT)
    # ============================================================
    def predict_mc_dropout(self, X, n_samples=20):
        """
        Enables dropout at inference for epistemic uncertainty.
        """
        results = []

        for _ in range(n_samples):
            preds = []
            for m in self.models:
                # Force training mode for dropout
                pred = m(X, training=True).numpy().flatten()
                preds.append(pred)

            results.append(np.mean(preds, axis=0))

        results = np.array(results)

        return np.mean(results, axis=0), np.std(results, axis=0)