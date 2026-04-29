from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin


class NaiveRegressor(BaseEstimator, RegressorMixin):
    """Predict zero return for every bar (price-stays-same baseline)."""

    def fit(self, X: object, y: object = None) -> "NaiveRegressor":
        return self

    def predict(self, X: object) -> np.ndarray:
        import pandas as pd
        n = len(X) if not isinstance(X, int) else X
        return np.zeros(n)


class EMARegressor(BaseEstimator, RegressorMixin):
    """
    Predict the exponentially-weighted mean of training targets.

    Fit stores one scalar (the EMA of y_train); predict returns that scalar
    for every test row.  Directional accuracy > NaiveRegressor when returns
    have a persistent drift.
    """

    def __init__(self, span: int = 20) -> None:
        self.span = span

    def fit(self, X: object, y: np.ndarray) -> "EMARegressor":
        import pandas as pd
        series = pd.Series(np.asarray(y))
        self.ema_value_: float = float(
            series.ewm(span=self.span, adjust=False).mean().iloc[-1]
        )
        return self

    def predict(self, X: object) -> np.ndarray:
        n = len(X)
        return np.full(n, self.ema_value_)
