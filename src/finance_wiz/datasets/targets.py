"""
Target construction.  All targets are horizon-aware: they look `horizon` bars
into the future.  The last `horizon` rows of any DataFrame will have NaN
targets and must be excluded before training.
"""
from __future__ import annotations

import pandas as pd


def forward_return(close: pd.Series, horizon: int = 1) -> pd.Series:
    """price[t+horizon] / price[t] - 1, aligned to index t."""
    return close.shift(-horizon) / close - 1


def forward_log_return(close: pd.Series, horizon: int = 1) -> pd.Series:
    import numpy as np
    return (close.shift(-horizon) / close).apply(np.log)


def build_Xy(
    features: pd.DataFrame,
    close: pd.Series,
    horizon: int = 1,
    target: str = "forward_return",
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Align features and target on the same index, dropping NaN rows from
    either side.  Returns (X, y) ready for an sklearn estimator.
    """
    if target == "forward_return":
        y = forward_return(close, horizon)
    elif target == "forward_log_return":
        y = forward_log_return(close, horizon)
    else:
        raise ValueError(f"Unknown target: {target!r}")

    combined = features.join(y.rename("__target__"), how="inner")
    combined = combined.dropna()
    return combined.drop(columns="__target__"), combined["__target__"]
