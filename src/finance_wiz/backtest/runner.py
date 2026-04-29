from __future__ import annotations

import warnings
import pandas as pd
import numpy as np
from sklearn.base import clone

from finance_wiz.backtest.metrics import compute_all
from finance_wiz.datasets.splits import walk_forward_splits


def backtest(
    model: object,
    X: pd.DataFrame,
    y: pd.Series,
    *,
    n_splits: int = 5,
    min_train_size: int = 100,
    horizon: int = 1,
    purge: int | None = None,
    embargo: int = 2,
) -> pd.DataFrame:
    """
    Walk-forward evaluation of any sklearn-compatible estimator.

    A fresh clone of `model` is fit on each training fold so no state bleeds
    between folds.  X and y are sliced as DataFrames/Series to preserve
    feature names (avoids sklearn feature-name warnings with LightGBM etc.).

    Returns a DataFrame with one row per fold plus a final 'mean' summary row.
    Columns: fold, train_size, test_size, mae, rmse, dir_acc, sharpe.
    """
    X_arr = X
    y_arr = y

    rows: list[dict] = []
    for fold_i, (tr, te) in enumerate(
        walk_forward_splits(
            len(X),
            n_splits=n_splits,
            min_train_size=min_train_size,
            horizon=horizon,
            purge=purge,
            embargo=embargo,
        )
    ):
        X_train = X_arr.iloc[tr]
        y_train = y_arr.iloc[tr]
        X_test = X_arr.iloc[te]
        y_test = y_arr.iloc[te]

        fold_model = clone(model)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="X does not have valid feature names")
            fold_model.fit(X_train, y_train)
            preds = fold_model.predict(X_test)

        metrics = compute_all(y_test.values, np.asarray(preds))
        rows.append(
            {
                "fold": fold_i,
                "train_size": len(tr),
                "test_size": len(te),
                **metrics,
            }
        )

    if not rows:
        raise ValueError("No folds were generated — dataset may be too small.")

    results = pd.DataFrame(rows)

    # Append a mean summary row across folds
    mean_row = results.mean(numeric_only=True).to_dict()
    mean_row["fold"] = "mean"
    results = pd.concat(
        [results, pd.DataFrame([mean_row])], ignore_index=True
    )
    results["fold"] = results["fold"].astype(str)

    return results
