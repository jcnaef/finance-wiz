from __future__ import annotations

import numpy as np


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def directional_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.sign(y_pred) == np.sign(y_true)))


def sharpe(y_true: np.ndarray, y_pred: np.ndarray, annualise: int = 252) -> float:
    """
    Long/short strategy Sharpe: go long when pred > 0, short when pred < 0.
    Annualised assuming `annualise` bars per year.
    """
    strategy_returns = np.sign(y_pred) * y_true
    std = strategy_returns.std()
    if std < 1e-12:
        return 0.0
    return float(strategy_returns.mean() / std * np.sqrt(annualise))


def compute_all(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "mae": mae(y_true, y_pred),
        "rmse": rmse(y_true, y_pred),
        "dir_acc": directional_accuracy(y_true, y_pred),
        "sharpe": sharpe(y_true, y_pred),
    }
