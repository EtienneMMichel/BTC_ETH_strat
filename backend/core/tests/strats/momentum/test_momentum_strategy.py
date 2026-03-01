import numpy as np
import pandas as pd
import pytest
from core.strats.momentum.strategy import MomentumStrategy


def monotone_up(n=300, seed=0):
    idx = pd.date_range("2020-01-01", periods=n, freq="D")
    return pd.DataFrame({
        "BTC": 10000 * np.exp(np.linspace(0, 1, n)),
        "ETH": 500 * np.exp(np.linspace(0, 0.8, n)),
    }, index=idx)


def monotone_down(n=300):
    idx = pd.date_range("2020-01-01", periods=n, freq="D")
    return pd.DataFrame({
        "BTC": 10000 * np.exp(-np.linspace(0, 1, n)),
        "ETH": 500 * np.exp(-np.linspace(0, 0.8, n)),
    }, index=idx)


def random_prices(n=300, seed=42):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2020-01-01", periods=n, freq="D")
    return pd.DataFrame({
        "BTC": 10000 * np.exp(np.cumsum(rng.standard_normal(n) * 0.02)),
        "ETH": 500 * np.exp(np.cumsum(rng.standard_normal(n) * 0.025)),
    }, index=idx)


def test_monotone_up_positive_weights():
    p = monotone_up()
    m = MomentumStrategy(horizons=[21, 63])
    m.fit(p)
    sig = m.predict_signal(p)
    assert sig["BTC"] > 0 and sig["ETH"] > 0


def test_monotone_down_negative_weights():
    p = monotone_down()
    m = MomentumStrategy(horizons=[21, 63])
    m.fit(p)
    sig = m.predict_signal(p)
    assert sig["BTC"] < 0 and sig["ETH"] < 0


def test_weights_in_bounds():
    p = random_prices()
    m = MomentumStrategy()
    m.fit(p)
    sig = m.predict_signal(p)
    assert -1.0 <= sig["BTC"] <= 1.0
    assert -1.0 <= sig["ETH"] <= 1.0


def test_total_notional_leq_one():
    p = random_prices()
    m = MomentumStrategy()
    m.fit(p)
    sig = m.predict_signal(p)
    assert abs(sig["BTC"]) + abs(sig["ETH"]) <= 1.0 + 1e-9


def test_predict_signal_keys():
    p = random_prices()
    m = MomentumStrategy()
    m.fit(p)
    sig = m.predict_signal(p)
    assert set(sig.keys()) == {"BTC", "ETH"}


def test_no_raise():
    p = random_prices()
    m = MomentumStrategy()
    m.fit(p)
    m.predict_signal(p)  # should not raise
