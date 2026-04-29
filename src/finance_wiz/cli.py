from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path

import typer

app = typer.Typer(name="fwiz", add_completion=False)

_RUNS_DIR = Path(os.environ.get("FWIZ_RUNS_DIR", Path.cwd() / "runs"))


@app.command()
def fetch(
    ticker: str = typer.Argument(..., help="Ticker symbol, e.g. KO"),
    interval: str = typer.Option("1d", help="Bar interval"),
    start: str = typer.Option("2010-01-01", help="Start date YYYY-MM-DD"),
    end: str = typer.Option("2026-01-01", help="End date YYYY-MM-DD"),
) -> None:
    """Download and cache OHLCV data for a ticker."""
    from finance_wiz.ingest.yfinance_source import YFinanceSource

    src = YFinanceSource()
    df = src.fetch(ticker, interval=interval, start=start, end=end)
    typer.echo(f"Fetched {len(df)} rows for {ticker} → data cache updated.")
    typer.echo(df.tail(3).to_string())


@app.command()
def run(
    config_path: Path = typer.Argument(..., help="Path to experiment YAML"),
) -> None:
    """Run a walk-forward backtest defined by a YAML experiment config."""
    import dataclasses
    import warnings

    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    from finance_wiz import config as cfg_module
    from finance_wiz.backtest.runner import backtest
    from finance_wiz.datasets.targets import build_Xy
    from finance_wiz.features.ta_features import TAFeaturizer
    from finance_wiz.ingest.yfinance_source import YFinanceSource
    from finance_wiz.registry import load_class

    cfg = cfg_module.load(config_path)
    typer.echo(f"Experiment : {cfg.name}")

    # ── Ingest ──────────────────────────────────────────────────────────────
    typer.echo(f"Fetching   : {cfg.data.ticker} {cfg.data.start} → {cfg.data.end}")
    df = YFinanceSource().fetch(
        cfg.data.ticker,
        interval=cfg.data.interval,
        start=cfg.data.start,
        end=cfg.data.end,
    )

    # ── Features ────────────────────────────────────────────────────────────
    featurizer = TAFeaturizer(
        sma_windows=tuple(cfg.features.sma_windows),
        ema_windows=tuple(cfg.features.ema_windows),
        rsi_window=cfg.features.rsi_window,
        atr_window=cfg.features.atr_window,
        bb_window=cfg.features.bb_window,
        macd_signal=cfg.features.macd_signal,
    )
    X_all = featurizer.fit_transform(df)
    X, y = build_Xy(X_all, df["close"], horizon=cfg.target.horizon, target=cfg.target.type)
    typer.echo(f"Dataset    : {len(X)} rows × {X.shape[1]} features")

    # ── Model ────────────────────────────────────────────────────────────────
    model_cls = load_class(cfg.model.cls)
    model = model_cls(**cfg.model.params)
    if cfg.model.scale_features:
        model = Pipeline([("scaler", StandardScaler()), ("model", model)])

    typer.echo(f"Model      : {model_cls.__name__} (scale={cfg.model.scale_features})")

    # ── Backtest ────────────────────────────────────────────────────────────
    typer.echo("Running walk-forward backtest…")
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        results = backtest(
            model,
            X,
            y,
            n_splits=cfg.backtest.n_splits,
            min_train_size=cfg.backtest.min_train_size,
            horizon=cfg.target.horizon,
            purge=cfg.backtest.purge,
            embargo=cfg.backtest.embargo,
        )

    typer.echo("\n" + results.to_string(index=False, float_format="%.5f"))

    # ── Persist results ──────────────────────────────────────────────────────
    _RUNS_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = f"{cfg.name}_{ts}"
    out_path = _RUNS_DIR / f"{run_id}.json"

    payload = {
        "run_id": run_id,
        "config": dataclasses.asdict(cfg),
        "results": results.to_dict(orient="records"),
    }
    out_path.write_text(json.dumps(payload, indent=2))
    typer.echo(f"\nResults    → {out_path}")


if __name__ == "__main__":
    app()
