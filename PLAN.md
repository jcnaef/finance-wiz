# finance_wiz — Build Plan

A local stock prediction engine with a swappable-algorithm pipeline.

## Constraints & goals

- **Hardware**: 16 GB RAM, 6 GB GPU.
- **Data source**: `yfinance` (initially).
- **Goal**: Pipeline where prediction algorithms can be swapped without touching ingest, features, or backtest code.
- **Language**: Python.
- **Package manager**: `uv`.

## Language decision

Python, chosen over Rust / Julia / C++ / TypeScript because:

- `yfinance` is native.
- The entire ML/quant ecosystem (pandas, polars, scikit-learn, PyTorch, statsmodels, lightgbm, darts) lives here.
- Heavy numerical work runs in C/CUDA under the hood, so Python's interpreter speed is a non-issue at our scale.
- 6 GB VRAM rules out large model training anyway, so we don't need a systems language.

## Model interface: scikit-learn estimator API

Standardize on `sklearn.base.BaseEstimator` (+ `RegressorMixin` or `ClassifierMixin`) as the contract every model implements.

Why:

- Native sklearn models work directly (`Ridge`, `RandomForestRegressor`, `MLPRegressor`, …).
- `lightgbm` and `xgboost` ship sklearn-compatible classes (`LGBMRegressor`, `XGBRegressor`).
- PyTorch models wrap cleanly via [`skorch`](https://skorch.readthedocs.io/).
- `statsmodels` / ARIMA / Prophet need a thin (~20-line) adapter to conform.
- Free machinery: `Pipeline`, `ColumnTransformer`, `GridSearchCV`, `cross_val_score`.

YAML experiment configs reference any class by import path:

```yaml
model:
  class: lightgbm.LGBMRegressor
  params: {n_estimators: 500, learning_rate: 0.05}
```

The runner does `import_class(cfg.model.class)(**cfg.model.params)`. Adding an algorithm = new file in `models/`, no other code changes.

## Repository layout

```
finance_wiz/
├── pyproject.toml
├── PLAN.md
├── config/
│   └── experiments/            # one YAML per experiment
├── src/finance_wiz/
│   ├── ingest/
│   │   ├── base.py             # DataSource protocol
│   │   └── yfinance_source.py  # yfinance + parquet cache
│   ├── features/
│   │   ├── base.py
│   │   └── ta_features.py      # causal-only TA indicators
│   ├── datasets/
│   │   ├── splits.py           # walk_forward_splits — sole splitting authority
│   │   └── targets.py          # forward_return etc., horizon-aware
│   ├── models/
│   │   ├── baselines.py        # naive, EMA
│   │   ├── tree.py             # lightgbm/xgboost wrappers (if needed)
│   │   ├── nn_lstm.py          # skorch-wrapped PyTorch
│   │   ├── nn_transformer.py
│   │   └── arima_adapter.py    # statsmodels → sklearn API
│   ├── backtest/
│   │   ├── runner.py           # walk-forward evaluation
│   │   └── metrics.py          # MAE, RMSE, directional acc, Sharpe
│   ├── registry.py             # name → class lookup
│   └── cli.py                  # `fwiz run config/experiments/foo.yaml`
├── data/                       # gitignored parquet cache
├── notebooks/
└── tests/
    └── test_no_future_leakage.py
```

## Stack

| Concern | Choice | Notes |
|---|---|---|
| Env / deps | `uv` | Fast, modern, lockfile-driven |
| DataFrame | `pandas` | Switch to `polars` only if we outgrow it |
| Storage | Parquet on disk | Keyed by `(ticker, interval, date_range)` |
| Classical ML | `scikit-learn`, `lightgbm`, `xgboost`, `statsmodels` | Often beat NNs on this kind of data |
| Deep learning | `PyTorch` via `skorch` | LSTM/GRU/Transformer fit in 6 GB for single-ticker / small panels |
| Optional model zoos | `darts`, `nixtla` | Wrap rather than reimplement |
| Backtesting | Custom walk-forward | Roll our own; `vectorbt` later if we want trade-level metrics |
| Experiment tracking | Structured JSON in `runs/` to start; `mlflow` once >20 experiments | |

## Time-series leakage — the key correctness concern

Standard sklearn CV (`KFold`, `train_test_split`) **shuffles rows**, which leaks the future into the past for time series. This is the single most common way quant ML backtests look brilliant and lose money live.

### Four leakage vectors and how we close them

**1. Splitting** — train must precede test in time.

- All splitting goes through `datasets/splits.py::walk_forward_splits`.
- Nothing else in the codebase calls `train_test_split` / `KFold` (convention enforced in review).
- Returns index arrays only — never touches the data, so it cannot leak features.
- Per-fold structure:

  ```
  |=========== train ===========|--purge--|-embargo-|=== test ===|
                                ^         ^         ^
                                target horizon ends |
                                          autocorr buffer
  ```

**2. Feature construction** — features must be computed *inside* the fold, after the split.

- Featurizers and scalers run as part of an `sklearn.Pipeline`:

  ```python
  pipe = Pipeline([
      ("features", TAFeaturizer(...)),
      ("scaler",   StandardScaler()),
      ("model",    LGBMRegressor(...)),
  ])
  ```

- For each fold: `pipe.fit(X_train, y_train)` then `pipe.predict(X_test)`. Scaler stats and any learned transforms see train only.
- Rolling indicators (SMA, RSI, …) are **causal by construction** (`df.rolling(20).mean()` looks only backward) and safe to precompute. `features/ta_features.py` documents this assumption explicitly so non-causal features can't be added quietly.

**3. Target horizon overlap** — for an N-step-forward target, the last N rows of train have targets overlapping the first N rows of test.

- Targets live in `datasets/targets.py` with horizon as a first-class config field.
- `walk_forward_splits` reads the horizon and enforces `purge >= horizon`; misconfiguration raises.
- An additional `embargo` (default ~2 rows) absorbs autocorrelation past the purge gap.

**4. Hyperparameter tuning** — the same rules apply recursively.

- `GridSearchCV(pipe, param_grid, cv=walk_forward_cv, ...)` — sklearn accepts any iterator of `(train_idx, test_idx)` tuples as `cv`.
- **Nested split**: outer fold for final evaluation, inner fold (on train only) for tuning. Same splitter both times.

### Reporting

Backtest emits **per-fold metrics** plus aggregates. Single averages hide regimes where one fold collapses; per-fold tables make that obvious.

### Leakage smoke test in CI

```python
def test_no_future_leakage():
    # Series where past is deterministic, future is random noise.
    # Better-than-chance test performance ⇒ leakage somewhere.
```

Cheap insurance against regressions when new featurizers land.

## Initial configuration

- **Universe**: single ticker — `KO` (Coca-Cola).
- **Target**: 1-day forward return (`price[t+1] / price[t] - 1`).
- **Horizon**: 1 day. Splitter purge defaults to ≥1, with a small embargo on top.

These are starting defaults; the YAML config layer makes them per-experiment overrides later.

## Build order

1. **Skeleton + ingest** — uv project, package layout, `yfinance_source.py` with parquet cache, CLI command that pulls a ticker and stores it.
2. **Features + dataset** — a few causal TA indicators, windowing into `(X, y)`.
3. **Two baseline models** — naive ("tomorrow = today") and `LGBMRegressor`. Both behind the sklearn API. Proves the swap point works.
4. **Backtest harness** — walk-forward evaluation that consumes any sklearn estimator and reports per-fold MAE/RMSE/directional accuracy/Sharpe.
5. **PyTorch model via skorch** — LSTM. Confirms the protocol holds for GPU-trained models.
6. **YAML-driven experiments + registry** — `fwiz run config/experiments/foo.yaml`, results land reproducibly under `runs/`.
7. **Leakage smoke test** added to CI as soon as the backtest harness exists.
