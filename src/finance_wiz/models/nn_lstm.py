"""
LSTM regressor conforming to the sklearn estimator API via skorch.

The core challenge: sklearn expects 2-D input (n_samples, n_features) but an
LSTM needs 3-D sequences (n_samples, seq_len, n_features).  We resolve this
by:

  fit()    — slides a window of length `seq_len` over X_train, producing
             (n_train - seq_len + 1) sequences.  The last (seq_len - 1) rows
             of X_train are stored as `context_` so predict() can build a
             full-length window for every test row without touching future
             data.

  predict() — prepends `context_` to X_test before windowing, so every test
              row gets exactly `seq_len` bars of history from the training
              period.  No leakage: context rows all precede the test window.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from sklearn.base import BaseEstimator, RegressorMixin
from skorch import NeuralNetRegressor


class _LSTMNet(nn.Module):
    def __init__(
        self,
        n_features: int,
        hidden_size: int = 64,
        n_layers: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            n_features,
            hidden_size,
            n_layers,
            dropout=dropout if n_layers > 1 else 0.0,
            batch_first=True,
        )
        self.head = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        return self.head(out[:, -1, :]).squeeze(-1)  # (batch,)


def _build_sequences(X: np.ndarray, seq_len: int) -> np.ndarray:
    """Slide a window of `seq_len` over rows of X → (n-seq_len+1, seq_len, n_features)."""
    n = len(X)
    return np.stack([X[i : i + seq_len] for i in range(n - seq_len + 1)], axis=0)


class LSTMRegressor(BaseEstimator, RegressorMixin):
    """Skorch-wrapped LSTM regressor; drop-in replacement for any sklearn estimator."""

    def __init__(
        self,
        seq_len: int = 20,
        hidden_size: int = 64,
        n_layers: int = 2,
        dropout: float = 0.1,
        max_epochs: int = 50,
        lr: float = 1e-3,
        batch_size: int = 64,
        device: str = "auto",
    ) -> None:
        self.seq_len = seq_len
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.dropout = dropout
        self.max_epochs = max_epochs
        self.lr = lr
        self.batch_size = batch_size
        self.device = device

    def _resolve_device(self) -> str:
        if self.device == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return self.device

    def fit(self, X: object, y: object) -> "LSTMRegressor":
        X_np = np.asarray(X, dtype=np.float32)
        y_np = np.asarray(y, dtype=np.float32)

        n_features = X_np.shape[1]

        X_seq = _build_sequences(X_np, self.seq_len)   # (n-seq+1, seq, feat)
        y_seq = y_np[self.seq_len - 1 :]               # align to last bar of each window

        # Stash training tail so predict() can form full windows for every test row
        self.context_ = X_np[-(self.seq_len - 1) :].copy()
        self.n_features_ = n_features

        self.net_ = NeuralNetRegressor(
            module=_LSTMNet,
            module__n_features=n_features,
            module__hidden_size=self.hidden_size,
            module__n_layers=self.n_layers,
            module__dropout=self.dropout,
            max_epochs=self.max_epochs,
            lr=self.lr,
            batch_size=self.batch_size,
            device=self._resolve_device(),
            verbose=0,
            train_split=None,
        )
        self.net_.fit(X_seq, y_seq)
        return self

    def predict(self, X: object) -> np.ndarray:
        X_np = np.asarray(X, dtype=np.float32)

        if self.seq_len > 1 and len(self.context_) > 0:
            X_with_context = np.concatenate([self.context_, X_np], axis=0)
        else:
            X_with_context = X_np

        X_seq = _build_sequences(X_with_context, self.seq_len)
        preds = self.net_.predict(X_seq)
        return np.asarray(preds).ravel()
