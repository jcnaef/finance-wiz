"""
Model registry: maps short names and full dotted paths to classes.

Usage:
    from finance_wiz.registry import load_class
    cls = load_class("lgbm")                         # short name
    cls = load_class("lightgbm.LGBMRegressor")       # full path
    cls = load_class("finance_wiz.models.nn_lstm.LSTMRegressor")
"""
from __future__ import annotations

import importlib

REGISTRY: dict[str, str] = {
    "naive":   "finance_wiz.models.baselines.NaiveRegressor",
    "ema":     "finance_wiz.models.baselines.EMARegressor",
    "lgbm":    "lightgbm.LGBMRegressor",
    "xgb":     "xgboost.XGBRegressor",
    "lstm":    "finance_wiz.models.nn_lstm.LSTMRegressor",
}


def load_class(path: str) -> type:
    """Return the class identified by `path` (short name or dotted import path)."""
    resolved = REGISTRY.get(path, path)
    module_path, class_name = resolved.rsplit(".", 1)
    module = importlib.import_module(module_path)
    return getattr(module, class_name)
