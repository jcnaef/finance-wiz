from __future__ import annotations

from typing import Protocol

import pandas as pd


class Featurizer(Protocol):
    def fit(self, X: pd.DataFrame, y: object = None) -> "Featurizer": ...
    def transform(self, X: pd.DataFrame) -> pd.DataFrame: ...
    def fit_transform(self, X: pd.DataFrame, y: object = None) -> pd.DataFrame: ...
