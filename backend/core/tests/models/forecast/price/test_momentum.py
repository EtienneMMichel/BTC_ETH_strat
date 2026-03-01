import numpy as np
import pandas as pd
import pytest
from core.models.forecast.price.momentum import TSMOMModel, MomentumModel


def monotone_up(n=300):
    idx = pd.date_range("2020-01-01", periods=n, freq="D")
    return pd.Series(100 * np.exp(np.linspace(0, 1, n)), index=idx)


def monotone_down(n=300):
    idx = pd.date_range("2020-01-01", periods=n, freq="D")
    return pd.Series(100 * np.exp(-np.linspace(0, 1, n)), index=idx)


def random_walk(n=1000, seed=42):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2020-01-01", periods=n, freq="D")
    return pd.Series(100 * np.exp(np.cumsum(rng.standard_normal(n) * 0.01)), index=idx)


def test_monotone_up_positive_signal():
    p = monotone_up()
    m = TSMOMModel(lookback=21)
    m.fit(p)
    pred = m.predict(p).dropna()
    assert (pred > 0).all(), "Monotone up prices should give positive signal"


def test_monotone_down_negative_signal():
    p = monotone_down()
    m = TSMOMModel(lookback=21)
    m.fit(p)
    pred = m.predict(p).dropna()
    assert (pred < 0).all(), "Monotone down prices should give negative signal"


def test_signal_index_alignment():
    p = random_walk()
    m = TSMOMModel()
    m.fit(p)
    pred = m.predict(p)
    assert pred.index.equals(p.index)


def test_random_walk_mean_near_zero():
    p = random_walk(n=2000)
    m = TSMOMModel(lookback=63)
    m.fit(p)
    pred = m.predict(p).dropna()
    assert abs(pred.mean()) < pred.std() * 0.5  # mean closer to 0 than 1 std


def test_multi_horizon_momentum():
    p = monotone_up(400)
    m = MomentumModel(horizons=[21, 63])
    m.fit(p)
    pred = m.predict(p).dropna()
    assert (pred > 0).all()
    assert pred.index.equals(p.index)
