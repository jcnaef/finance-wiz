"""
Experiment config loader.

YAML schema
-----------
name: string

data:
  ticker: str          # e.g. KO
  interval: str        # e.g. 1d
  start: str           # YYYY-MM-DD
  end: str             # YYYY-MM-DD

features:              # all optional — defaults match TAFeaturizer.__init__
  sma_windows: [5, 10, 20]
  ema_windows: [12, 26]
  rsi_window: 14
  atr_window: 14
  bb_window: 20
  macd_signal: 9

target:
  type: forward_return  # or forward_log_return
  horizon: 1

backtest:
  n_splits: 5
  min_train_size: 252
  purge: null           # defaults to horizon
  embargo: 2

model:
  class: lightgbm.LGBMRegressor   # short name or full dotted path
  scale_features: false            # wrap in StandardScaler pipeline
  params: {}                       # passed as **kwargs to the constructor
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class DataConfig:
    ticker: str = "KO"
    interval: str = "1d"
    start: str = "2010-01-01"
    end: str = "2026-01-01"


@dataclass
class FeaturesConfig:
    sma_windows: list[int] = field(default_factory=lambda: [5, 10, 20])
    ema_windows: list[int] = field(default_factory=lambda: [12, 26])
    rsi_window: int = 14
    atr_window: int = 14
    bb_window: int = 20
    macd_signal: int = 9


@dataclass
class TargetConfig:
    type: str = "forward_return"
    horizon: int = 1


@dataclass
class BacktestConfig:
    n_splits: int = 5
    min_train_size: int = 252
    purge: int | None = None
    embargo: int = 2


@dataclass
class ModelConfig:
    cls: str = "naive"
    scale_features: bool = False
    params: dict[str, Any] = field(default_factory=dict)


@dataclass
class ExperimentConfig:
    name: str
    data: DataConfig
    features: FeaturesConfig
    target: TargetConfig
    backtest: BacktestConfig
    model: ModelConfig


def load(path: str | Path) -> ExperimentConfig:
    raw = yaml.safe_load(Path(path).read_text())

    data = DataConfig(**raw.get("data", {}))
    features = FeaturesConfig(**raw.get("features", {}))
    target = TargetConfig(**raw.get("target", {}))
    backtest = BacktestConfig(**raw.get("backtest", {}))

    model_raw = raw.get("model", {})
    model = ModelConfig(
        cls=model_raw.get("class", "naive"),
        scale_features=model_raw.get("scale_features", False),
        params=model_raw.get("params", {}),
    )

    return ExperimentConfig(
        name=raw.get("name", Path(path).stem),
        data=data,
        features=features,
        target=target,
        backtest=backtest,
        model=model,
    )
