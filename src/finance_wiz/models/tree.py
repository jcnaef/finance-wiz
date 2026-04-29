"""
Tree-based models.  LGBMRegressor and XGBRegressor already implement the
sklearn estimator API natively, so this module re-exports them under a
stable import path and documents the recommended hyper-parameter starting
points for this project.
"""
from __future__ import annotations

from lightgbm import LGBMRegressor
from xgboost import XGBRegressor

__all__ = ["LGBMRegressor", "XGBRegressor"]
