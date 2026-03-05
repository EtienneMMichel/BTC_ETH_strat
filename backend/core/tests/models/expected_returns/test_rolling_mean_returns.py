import numpy as np
import pandas as pd
import pytest
from core.models.expected_returns.rolling_mean import RollingMeanReturns


def make_prices(n=120, seed=0):
    rng = np.random.default_rng(seed)
    btc = np.cumprod(1 + rng.normal(0.001, 0.02, n))
    eth = np.cumprod(1 + rng.normal(0.001, 0.025, n))
    idx = pd.date_range("2020-01-01", periods=n, freq="D", tz="UTC")
    return pd.DataFrame({"BTC": btc, "ETH": eth}, index=idx)


def test_rolling_mean_columns():
    prices = make_prices()
    m = RollingMeanReturns()
    m.fit(prices)
    pred = m.predict(prices)
    assert set(pred.columns) == {"BTC", "ETH"}


def test_rolling_mean_index():
    prices = make_prices()
    m = RollingMeanReturns()
    m.fit(prices)
    pred = m.predict(prices)
    assert pred.index.equals(prices.index)


def test_rolling_mean_no_lookahead():
    """Changing the last price row must not alter any earlier prediction."""
    prices = make_prices(100)
    m = RollingMeanReturns()
    m.fit(prices)
    pred_orig = m.predict(prices).iloc[:-1].copy()

    prices2 = prices.copy()
    prices2.iloc[-1] *= 100
    pred_new = m.predict(prices2).iloc[:-1]

    pd.testing.assert_frame_equal(pred_orig, pred_new)


def test_rolling_mean_positive_drift():
    """Series with consistently positive log-returns → positive rolling mean."""
    n = 200
    idx = pd.date_range("2020-01-01", periods=n, freq="D", tz="UTC")
    prices = pd.DataFrame({
        "BTC": np.cumprod(1 + np.full(n, 0.01)),
        "ETH": np.cumprod(1 + np.full(n, 0.01)),
    }, index=idx)
    m = RollingMeanReturns(window=30)
    m.fit(prices)
    pred = m.predict(prices).dropna()
    assert (pred["BTC"] > 0).all()
    assert (pred["ETH"] > 0).all()


def test_rolling_mean_short_series():
    prices = make_prices(10)
    m = RollingMeanReturns(window=63, min_periods=2)
    m.fit(prices)
    pred = m.predict(prices)
    assert len(pred) == 10


def test_rolling_mean_finite_output():
    prices = make_prices(150)
    m = RollingMeanReturns()
    m.fit(prices)
    pred = m.predict(prices).dropna()
    assert np.isfinite(pred.values).all()
