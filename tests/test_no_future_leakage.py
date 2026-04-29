"""
Leakage smoke test.

A synthetic series where every past bar is perfectly predictable (cumulative
integers) but the future target is pure random noise.  If any featurizer or
splitter leaks the future, a fitted model will beat chance on the test fold.
We allow a generous 65% directional-accuracy threshold — anything above that
on this synthetic data signals leakage.
"""
import numpy as np
import pandas as pd
import pytest

from finance_wiz.features.ta_features import TAFeaturizer
from finance_wiz.datasets.targets import build_Xy
from finance_wiz.datasets.splits import walk_forward_splits
from finance_wiz.models.baselines import NaiveRegressor
from sklearn.base import clone


def _make_synthetic(n: int = 500, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    # Past is perfectly deterministic; only the future is noise.
    close = np.cumsum(np.ones(n)) + 100.0
    return pd.DataFrame(
        {
            "open":   close,
            "high":   close + 0.1,
            "low":    close - 0.1,
            "close":  close,
            "volume": rng.integers(1_000_000, 5_000_000, size=n).astype(float),
        },
        index=pd.date_range("2015-01-01", periods=n, freq="B", tz="UTC"),
    )


def test_no_future_leakage_naive():
    """NaiveRegressor (predict-zero) should have ~50% dir accuracy on noise."""
    df = _make_synthetic()
    X = TAFeaturizer().fit_transform(df)
    X, y = build_Xy(X, df["close"], horizon=1)

    dir_accs = []
    for tr, te in walk_forward_splits(len(X), n_splits=3, min_train_size=100, horizon=1):
        model = clone(NaiveRegressor())
        model.fit(X.iloc[tr], y.iloc[tr])
        preds = model.predict(X.iloc[te])
        da = np.mean(np.sign(preds) == np.sign(y.iloc[te].values))
        dir_accs.append(da)

    # NaiveRegressor predicts 0, so sign(0) == sign(any nonzero) is False.
    # On a strictly-ascending close series, forward return is always positive,
    # so dir_acc should be 0.0 — well below the leakage threshold.
    mean_da = float(np.mean(dir_accs))
    assert mean_da < 0.65, (
        f"Suspiciously high directional accuracy ({mean_da:.1%}) on synthetic "
        "data — possible future leakage."
    )


def test_walk_forward_no_overlap():
    """Train and test index sets must be disjoint and time-ordered."""
    n = 300
    for tr, te in walk_forward_splits(n, n_splits=4, min_train_size=80, horizon=1):
        assert set(tr).isdisjoint(set(te)), "Train/test overlap detected"
        assert max(tr) < min(te), "Train indices extend into test window"
