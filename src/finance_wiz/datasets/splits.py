"""
Walk-forward (expanding-window) cross-validation for time series.

Split structure per fold:

    |=========== train ===========|--purge--|-embargo-|=== test ===|
                                  ^         ^         ^
                       target horizon ends  |
                                   autocorr buffer

Rules enforced:
  - purge >= horizon  (raises ValueError otherwise)
  - train end < test start (guaranteed by construction)
  - No row appears in both train and test

The iterator yields plain integer-position index arrays (iloc-compatible),
not label-based indices, so it works directly as the `cv` argument to
GridSearchCV even when the DataFrame has a non-integer index.
"""
from __future__ import annotations

from collections.abc import Iterator


def walk_forward_splits(
    n: int,
    *,
    n_splits: int = 5,
    min_train_size: int = 252,
    horizon: int = 1,
    purge: int | None = None,
    embargo: int = 2,
) -> Iterator[tuple[list[int], list[int]]]:
    """
    Yield (train_indices, test_indices) for an expanding-window walk-forward CV.

    Parameters
    ----------
    n:              Total number of rows in the dataset.
    n_splits:       Number of folds.
    min_train_size: Minimum rows in the first training fold.
    horizon:        Target look-ahead in bars (sets minimum purge).
    purge:          Rows to drop between train end and test start.
                    Defaults to `horizon`; must be >= horizon.
    embargo:        Additional rows to skip after purge for autocorrelation.
    """
    effective_purge = purge if purge is not None else horizon
    if effective_purge < horizon:
        raise ValueError(
            f"purge ({effective_purge}) must be >= horizon ({horizon})"
        )

    gap = effective_purge + embargo

    # Determine test-fold size: split the rows after min_train_size evenly.
    available = n - min_train_size - gap
    if available <= 0:
        raise ValueError(
            f"Dataset too small: n={n}, min_train_size={min_train_size}, gap={gap}"
        )

    test_size = max(1, available // n_splits)

    for fold in range(n_splits):
        # Expanding train: grows with each fold.
        train_end = min_train_size + fold * test_size
        test_start = train_end + gap
        test_end = test_start + test_size

        if test_end > n:
            break

        train_idx = list(range(0, train_end))
        test_idx = list(range(test_start, test_end))
        yield train_idx, test_idx


def walk_forward_cv(
    n: int,
    **kwargs: object,
) -> list[tuple[list[int], list[int]]]:
    """Materialise all folds as a list (compatible with sklearn's cv= argument)."""
    return list(walk_forward_splits(n, **kwargs))
