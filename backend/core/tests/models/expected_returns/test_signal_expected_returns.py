import numpy as np
import pandas as pd
import pytest
from core.models.expected_returns.signal import SignalExpectedReturns
from core.models.forecast.price.momentum import TSMOMModel, MomentumModel


def make_prices(n=300, seed=42):
    rng = np.random.default_rng(seed)
    btc = np.cumprod(1 + rng.normal(0.0005, 0.02, n))
    eth = np.cumprod(1 + rng.normal(0.0005, 0.025, n))
    idx = pd.date_range("2020-01-01", periods=n, freq="D", tz="UTC")
    return pd.DataFrame({"BTC": btc, "ETH": eth}, index=idx)


def test_signal_er_columns():
    prices = make_prices()
    m = SignalExpectedReturns()
    m.fit(prices)
    pred = m.predict(prices)
    assert set(pred.columns) == {"BTC", "ETH"}


def test_signal_er_index():
    prices = make_prices()
    m = SignalExpectedReturns()
    m.fit(prices)
    pred = m.predict(prices)
    assert pred.index.equals(prices.index)


def test_signal_er_no_lookahead():
    """Changing the last price must not alter any earlier prediction."""
    prices = make_prices(300)
    m = SignalExpectedReturns(model_cls=TSMOMModel, model_kwargs={"lookback": 60})
    m.fit(prices)
    pred_orig = m.predict(prices).iloc[:-1].copy()

    prices2 = prices.copy()
    prices2.iloc[-1] *= 50
    pred_new = m.predict(prices2).iloc[:-1]

    pd.testing.assert_frame_equal(pred_orig, pred_new)


def test_signal_er_default_model_cls():
    m = SignalExpectedReturns()
    assert m._model_cls is TSMOMModel


def test_signal_er_custom_model_cls():
    m = SignalExpectedReturns(model_cls=MomentumModel)
    assert m._model_cls is MomentumModel


def test_signal_er_finite_output():
    prices = make_prices(300)
    m = SignalExpectedReturns()
    m.fit(prices)
    pred = m.predict(prices).dropna()
    assert np.isfinite(pred.values).all()


def test_signal_er_positive_trend():
    """Monotone up-trending series → positive signal → positive expected return."""
    n = 300
    idx = pd.date_range("2020-01-01", periods=n, freq="D", tz="UTC")
    prices = pd.DataFrame({
        "BTC": np.cumprod(1 + np.full(n, 0.005)),
        "ETH": np.cumprod(1 + np.full(n, 0.005)),
    }, index=idx)
    m = SignalExpectedReturns(model_cls=TSMOMModel, model_kwargs={"lookback": 60})
    m.fit(prices)
    pred = m.predict(prices).dropna().iloc[60:]  # skip warm-up
    assert (pred["BTC"] > 0).mean() > 0.8
    assert (pred["ETH"] > 0).mean() > 0.8


def test_signal_er_scale():
    prices = make_prices(300)
    m1 = SignalExpectedReturns(scale=1.0)
    m2 = SignalExpectedReturns(scale=2.0)
    m1.fit(prices)
    m2.fit(prices)
    pred1 = m1.predict(prices).dropna()
    pred2 = m2.predict(prices).dropna()
    pd.testing.assert_frame_equal(pred2, pred1 * 2.0, check_names=False)
