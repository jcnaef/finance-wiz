from __future__ import annotations

from typing import Protocol

import pandas as pd


class DataSource(Protocol):
    def fetch(
        self,
        ticker: str,
        interval: str,
        start: str,
        end: str,
    ) -> pd.DataFrame:
        """Return OHLCV DataFrame indexed by UTC datetime."""
        ...
