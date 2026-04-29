"""
Causal-only technical analysis features.

Every indicator here is computed with a strictly backward-looking window so
features built on row t use only information available at time t.  No
min_periods trick that sneaks future data in; each indicator drops NaN rows
that lack enough history rather than filling with forward data.

Fit is stateless — all transforms are deterministic functions of the input.
The sklearn Pipeline still wraps this so downstream scalers (StandardScaler)
learn their stats only from the training fold.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class TAFeaturizer(BaseEstimator, TransformerMixin):
    """Compute a fixed set of causal TA indicators from OHLCV data."""

    def __init__(
        self,
        sma_windows: tuple[int, ...] = (5, 10, 20),
        ema_windows: tuple[int, ...] = (12, 26),
        rsi_window: int = 14,
        atr_window: int = 14,
        bb_window: int = 20,
        macd_signal: int = 9,
    ) -> None:
        self.sma_windows = sma_windows
        self.ema_windows = ema_windows
        self.rsi_window = rsi_window
        self.atr_window = atr_window
        self.bb_window = bb_window
        self.macd_signal = macd_signal

    # fit is stateless; exists only to satisfy the sklearn API
    def fit(self, X: pd.DataFrame, y: object = None) -> "TAFeaturizer":
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        close = X["close"]
        high = X["high"]
        low = X["low"]
        volume = X["volume"]

        feats: dict[str, pd.Series] = {}

        # ── Price-ratio features (dimensionless, robust to level changes) ──
        feats["hl_spread"] = (high - low) / close
        feats["co_return"] = (close - X["open"]) / X["open"]
        feats["log_return"] = np.log(close / close.shift(1))

        # ── SMAs as ratio to close ──
        for w in self.sma_windows:
            sma = close.rolling(w).mean()
            feats[f"sma{w}_ratio"] = close / sma - 1

        # ── EMAs as ratio to close ──
        for w in self.ema_windows:
            ema = close.ewm(span=w, adjust=False).mean()
            feats[f"ema{w}_ratio"] = close / ema - 1

        # ── MACD histogram (fast_ema - slow_ema, normalised by close) ──
        fast_ema = close.ewm(span=self.ema_windows[0], adjust=False).mean()
        slow_ema = close.ewm(span=self.ema_windows[-1], adjust=False).mean()
        macd_line = fast_ema - slow_ema
        signal_line = macd_line.ewm(span=self.macd_signal, adjust=False).mean()
        feats["macd_hist"] = (macd_line - signal_line) / close

        # ── RSI ──
        feats["rsi"] = _rsi(close, self.rsi_window)

        # ── ATR normalised by close ──
        feats["atr_ratio"] = _atr(high, low, close, self.atr_window) / close

        # ── Bollinger Band width and %B ──
        sma_bb = close.rolling(self.bb_window).mean()
        std_bb = close.rolling(self.bb_window).std()
        feats["bb_width"] = (2 * std_bb) / sma_bb
        feats["bb_pct"] = (close - (sma_bb - 2 * std_bb)) / (4 * std_bb)

        # ── Volume z-score (rolling, causal) ──
        vol_mean = volume.rolling(20).mean()
        vol_std = volume.rolling(20).std()
        feats["volume_z"] = (volume - vol_mean) / (vol_std + 1e-9)

        out = pd.DataFrame(feats, index=X.index)
        return out.dropna()


# ── helpers ─────────────────────────────────────────────────────────────────

def _rsi(close: pd.Series, window: int) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(com=window - 1, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(com=window - 1, adjust=False).mean()
    rs = gain / (loss + 1e-9)
    return 100 - 100 / (1 + rs)


def _atr(high: pd.Series, low: pd.Series, close: pd.Series, window: int) -> pd.Series:
    prev_close = close.shift(1)
    tr = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1
    ).max(axis=1)
    return tr.ewm(com=window - 1, adjust=False).mean()
