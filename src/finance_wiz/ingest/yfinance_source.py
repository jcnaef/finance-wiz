from __future__ import annotations

from pathlib import Path

import pandas as pd
import yfinance as yf


import os

_CACHE_DIR = Path(os.environ.get("FWIZ_DATA_DIR", Path.cwd() / "data"))


def _cache_path(ticker: str, interval: str, start: str, end: str) -> Path:
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    safe = f"{ticker}_{interval}_{start}_{end}".replace("/", "-")
    return _CACHE_DIR / f"{safe}.parquet"


class YFinanceSource:
    """yfinance-backed DataSource with parquet cache."""

    def fetch(
        self,
        ticker: str,
        interval: str = "1d",
        start: str = "2010-01-01",
        end: str = "2026-01-01",
    ) -> pd.DataFrame:
        path = _cache_path(ticker, interval, start, end)
        if path.exists():
            return pd.read_parquet(path)

        raw = yf.download(
            ticker,
            start=start,
            end=end,
            interval=interval,
            auto_adjust=True,
            progress=False,
        )

        if raw.empty:
            raise ValueError(f"yfinance returned no data for {ticker!r}")

        # Flatten MultiIndex columns produced by yfinance when a single ticker
        # is passed as a list; normalise to lowercase.
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = [col[0].lower() for col in raw.columns]
        else:
            raw.columns = [c.lower() for c in raw.columns]

        raw.index = pd.to_datetime(raw.index, utc=True)
        raw.index.name = "datetime"
        raw = raw.sort_index()

        raw.to_parquet(path)
        return raw
