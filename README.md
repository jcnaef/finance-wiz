# finance_wiz

A local stock prediction engine with a **swappable-algorithm pipeline**. Swap models by changing one line in a YAML config — ingest, features, and backtest code stay untouched.

## Features

- Walk-forward backtester with purge + embargo gaps to prevent look-ahead bias
- Sklearn-compatible model interface — native sklearn, LightGBM, XGBoost, and PyTorch (via skorch) all plug in the same way
- YAML-driven experiments with reproducible JSON result artifacts
- Parquet-cached OHLCV data via `yfinance`
- Causal-only TA indicators (SMA, EMA, RSI, ATR, Bollinger Bands, MACD)

## Requirements

- Python 3.11–3.12
- [uv](https://github.com/astral-sh/uv)
- CUDA optional (CPU training works; 6 GB VRAM sufficient for LSTM)

## Installation

```bash
git clone <https://github.com/jcnaef/finance-wiz.git>
cd finance_wiz
uv sync
```

## Usage

### Fetch data

```bash
fwiz fetch KO --start 2015-01-01 --end 2024-01-01
```

### Run an experiment

```bash
fwiz run config/experiments/ko_lgbm.yaml
fwiz run config/experiments/ko_lstm.yaml
fwiz run config/experiments/ko_naive.yaml
```

Results are written to `runs/<name>_<timestamp>.json`.

## Results

KO 2015–2024 · 5 walk-forward folds · metrics are means across folds

### 1-day return (interval=1d)

| Model    |   MAE    |   RMSE   | Dir Acc | Sharpe |
|----------|----------|----------|---------|--------|
| Naive    | 0.0078   | 0.0113   |   0.8%  |  0.00  |
| EMA      | 0.0080   | 0.0115   |  51.3%  |  0.38  |
| LightGBM | 0.0087   | 0.0122   |  49.8%  |  0.11  |
| XGBoost  | 0.0087   | 0.0122   |  49.9%  |  0.08  |
| LSTM     | 0.0101   | 0.0138   |  49.2%  | -0.17  |

### 5-day return (interval=5d)

| Model   |   MAE    |   RMSE   | Dir Acc | Sharpe |
|---------|----------|----------|---------|--------|
| EMA     | 0.0163   | 0.0229   |  53.7%  |  1.38  |
| LSTM    | 0.0205   | 0.0308   |  55.6%  |  0.90  |
| XGBoost | 0.0224   | 0.0327   |  48.4%  | -0.18  |

Dir Acc = fraction of test-set bars where predicted direction matched actual.

## Experiment config

```yaml
name: ko_lgbm

data:
  ticker: KO
  interval: 1d
  start: "2015-01-01"
  end: "2024-01-01"

features:
  sma_windows: [5, 10, 20]
  ema_windows: [12, 26]

target:
  type: forward_return
  horizon: 1

backtest:
  n_splits: 5
  min_train_size: 252
  embargo: 2

model:
  class: lgbm          # short name or full dotted path
  scale_features: false
  params:
    n_estimators: 200
    learning_rate: 0.05
```

**Built-in model aliases:**

| Alias  | Class                                          |
|--------|------------------------------------------------|
| naive  | `finance_wiz.models.baselines.NaiveRegressor`  |
| ema    | `finance_wiz.models.baselines.EMARegressor`    |
| lgbm   | `lightgbm.LGBMRegressor`                       |
| xgb    | `xgboost.XGBRegressor`                         |
| lstm   | `finance_wiz.models.nn_lstm.LSTMRegressor`     |

You can also pass any fully-qualified class path (e.g. `sklearn.linear_model.Ridge`) without registering it.

## Project layout

```
finance_wiz/
├── config/experiments/     # one YAML per experiment
├── src/finance_wiz/
│   ├── ingest/             # yfinance + parquet cache
│   ├── features/           # causal TA indicators
│   ├── datasets/           # walk-forward splits, target construction
│   ├── models/             # baselines, LightGBM tree wrapper, LSTM
│   ├── backtest/           # walk-forward runner + metrics
│   ├── registry.py         # short name → class lookup
│   └── cli.py              # fwiz CLI
├── data/                   # gitignored parquet cache
├── runs/                   # experiment result JSON artifacts
└── tests/
    └── test_no_future_leakage.py
```

## Leakage prevention

Time-series backtests are easy to overfit by accident. Four vectors are closed:

1. **Splitting** — all splits go through `walk_forward_splits`; `KFold`/`train_test_split` are never used.
2. **Features** — featurizers and scalers are fit inside each fold via an sklearn `Pipeline`, so no test-set statistics bleed into training.
3. **Target horizon** — a `purge >= horizon` gap between train and test is enforced at config load time.
4. **Hyperparameter tuning** — `GridSearchCV` uses the same walk-forward splitter, with a nested outer/inner fold structure.

A smoke test in `tests/test_no_future_leakage.py` guards against regressions.

## Running tests

```bash
uv run pytest
```
