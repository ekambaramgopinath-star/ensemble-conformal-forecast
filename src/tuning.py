"""
Hyperparameter tuning stub using Optuna.

Install optuna first: pip install optuna
Then implement the objective function with your model training logic.
"""

# import optuna
# from src.models import EnsembleWrapper
# from src.processor import DataProcessor
# from src.conformal import get_residuals, apply_enbpi
# from src.evaluation import rmse


def objective(trial):
    """
    Optuna objective: suggest hyperparameters, train LSTM, return val RMSE.
    Fill in training logic before running.
    """
    lookback = trial.suggest_int("lookback", 5, 60)
    lr = trial.suggest_float("lr", 1e-5, 1e-2, log=True)

    # TODO: build DataProcessor(window=lookback), train EnsembleWrapper,
    # compute residuals and return validation RMSE.
    raise NotImplementedError("Implement training loop before running tuning.")


# To run:
# study = optuna.create_study(direction="minimize")
# study.optimize(objective, n_trials=50)
