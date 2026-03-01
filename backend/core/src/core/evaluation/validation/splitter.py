from __future__ import annotations
from typing import Iterator
import pandas as pd


def walk_forward_splits(
    index: pd.DatetimeIndex,
    min_train: int,
    test_size: int,
    gap: int = 1,
    expanding: bool = True,
    rolling_train: int | None = None,
) -> Iterator[tuple[pd.DatetimeIndex, pd.DatetimeIndex]]:
    """
    Yield (train_idx, test_idx) pairs for walk-forward cross-validation.

    Parameters
    ----------
    index       : full DatetimeIndex of the dataset
    min_train   : minimum number of bars in the training set
    test_size   : number of bars per test fold
    gap         : bars to skip between train end and test start (default 1, prevents leakage)
    expanding   : True = expanding window; False = rolling window of fixed length (rolling_train)
    rolling_train: size of rolling training window (only used if expanding=False)
    """
    n = len(index)
    train_end = min_train  # exclusive upper bound (i.e., train = index[:train_end])

    while train_end + gap + test_size <= n:
        test_start = train_end + gap
        test_end = test_start + test_size

        if expanding:
            train_idx = index[:train_end]
        else:
            w = rolling_train or min_train
            train_idx = index[max(0, train_end - w):train_end]

        test_idx = index[test_start:test_end]
        yield train_idx, test_idx

        train_end += test_size  # advance by one test fold
