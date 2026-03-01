import pandas as pd
import pytest
from core.evaluation.validation.splitter import walk_forward_splits


def make_index(n=200):
    return pd.date_range("2020-01-01", periods=n, freq="D")


def test_no_overlap_train_test():
    idx = make_index(200)
    for train_idx, test_idx in walk_forward_splits(idx, min_train=100, test_size=20, gap=1):
        assert len(train_idx.intersection(test_idx)) == 0, "Train and test must not overlap"


def test_gap_respected():
    idx = make_index(200)
    for train_idx, test_idx in walk_forward_splits(idx, min_train=100, test_size=20, gap=5):
        # test must start at least gap bars after train ends
        train_end = train_idx[-1]
        test_start = test_idx[0]
        delta = (test_start - train_end).days
        assert delta >= 5


def test_expanding_window():
    idx = make_index(300)
    folds = list(walk_forward_splits(idx, min_train=100, test_size=30, gap=1, expanding=True))
    # Each fold's training set should be larger than the previous
    for i in range(1, len(folds)):
        assert len(folds[i][0]) > len(folds[i-1][0])


def test_rolling_window_fixed_size():
    idx = make_index(300)
    folds = list(walk_forward_splits(
        idx, min_train=100, test_size=30, gap=1,
        expanding=False, rolling_train=100,
    ))
    for train_idx, _ in folds:
        assert len(train_idx) == 100


def test_correct_fold_count():
    idx = make_index(200)
    folds = list(walk_forward_splits(idx, min_train=100, test_size=20, gap=1))
    # After min_train=100, remaining = 100 bars. With test_size=20 and gap=1:
    # fold 1: train[:100], test[101:121]  (train_end=100, test_start=101, test_end=121)
    # fold 2: train[:120], test[121:141]
    # fold 3: train[:140], test[141:161]
    # fold 4: train[:160], test[161:181]
    # fold 5: train[:180], test[181:201] — test_end=201 > 200, so stops
    # Actually: train_end starts at 100, advances by test_size=20 each fold
    # fold k: train_end = 100 + (k-1)*20; test_start = train_end+1; test_end = test_start+20
    # Stop when test_end > 200: 100 + (k-1)*20 + 1 + 20 <= 200 → (k-1)*20 <= 79 → k <= 4.95 → k=4
    assert len(folds) == 4


def test_test_size_correct():
    idx = make_index(300)
    for _, test_idx in walk_forward_splits(idx, min_train=100, test_size=25, gap=1):
        assert len(test_idx) == 25
