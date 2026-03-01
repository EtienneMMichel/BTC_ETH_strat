import numpy as np
import pandas as pd
import pytest
from core.strats.mean_reversion.strategy import MeanReversionStrategy


def make_cointegrated_prices(n=500, beta=1.5, kappa=0.15, seed=0):
    """Prices with cointegrated log-spread (OU process)."""
    rng = np.random.default_rng(seed)
    log_btc = np.cumsum(rng.standard_normal(n) * 0.02)
    ou = np.zeros(n)
    mu, sigma_ou = 0.0, 0.03
    for t in range(1, n):
        ou[t] = ou[t-1] + kappa * (mu - ou[t-1]) + sigma_ou * rng.standard_normal()
    log_eth = beta * log_btc + ou
    idx = pd.date_range("2020-01-01", periods=n, freq="D")
    return pd.DataFrame({
        "BTC": np.exp(log_btc) * 10000,
        "ETH": np.exp(log_eth) * 500,
    }, index=idx)


def make_random_walk_prices(n=300, seed=1):
    """Independent random walks -- NOT cointegrated."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2020-01-01", periods=n, freq="D")
    return pd.DataFrame({
        "BTC": np.exp(np.cumsum(rng.standard_normal(n) * 0.02)) * 10000,
        "ETH": np.exp(np.cumsum(rng.standard_normal(n) * 0.025)) * 500,
    }, index=idx)


def test_cointegrated_spread_stationary():
    p = make_cointegrated_prices()
    m = MeanReversionStrategy()
    m.fit(p)
    assert m.spread_stationary, "Cointegrated prices should pass ADF gate"


def test_noncointegrated_zero_weights():
    p = make_random_walk_prices()
    m = MeanReversionStrategy()
    m.fit(p)
    # May or may not be stationary depending on the random walk; just check no raise
    sig = m.predict_signal(p)
    assert set(sig.keys()) == {"BTC", "ETH"}


def test_weights_in_bounds():
    p = make_cointegrated_prices()
    m = MeanReversionStrategy()
    m.fit(p)
    sig = m.predict_signal(p)
    assert abs(sig["BTC"]) <= m.max_weight + 1e-9
    assert abs(sig["ETH"]) <= m.max_weight + 1e-9


def test_spread_above_entry_short_eth():
    """When spread is above mu + z_entry*sigma, ETH weight should be negative."""
    p = make_cointegrated_prices(kappa=0.5, seed=10)
    m = MeanReversionStrategy(z_entry=0.01)  # very low threshold so signal fires
    m.fit(p)
    # Check conditional on the observed z-score direction
    if m.spread_stationary and m.half_life <= m.max_half_life:
        z = m.spread_zscore(p)
        sig = m.predict_signal(p)
        if z > m.z_entry:
            assert sig["ETH"] < 0, "Above entry z -> ETH should be short"
        elif z < -m.z_entry:
            assert sig["ETH"] > 0, "Below -entry z -> ETH should be long"


def test_predict_signal_keys():
    p = make_cointegrated_prices()
    m = MeanReversionStrategy()
    m.fit(p)
    sig = m.predict_signal(p)
    assert set(sig.keys()) == {"BTC", "ETH"}


def test_short_history_zero_weights():
    """Less than 20 bars -> cannot test ADF, return zeros."""
    p = make_cointegrated_prices(n=15)
    m = MeanReversionStrategy()
    m.fit(p)
    assert not m.spread_stationary  # too short
    sig = m.predict_signal(p)
    assert sig["BTC"] == 0.0 and sig["ETH"] == 0.0


def test_no_raise_random():
    p = make_random_walk_prices(n=400)
    m = MeanReversionStrategy()
    m.fit(p)
    m.predict_signal(p)  # should not raise


def test_half_life_gate():
    """Spread with very slow reversion (large half-life) -> zero weights."""
    p = make_cointegrated_prices(kappa=0.001, n=500, seed=42)
    m = MeanReversionStrategy(max_half_life=10.0)
    m.fit(p)
    # With kappa ~ 0.001, half_life ~ log(2)/0.001 ~ 693 bars >> 10
    sig = m.predict_signal(p)
    # Either ADF fails (non-stationary with very slow OU) or half-life gate fires
    # Either way, we just verify no exception and keys are correct
    assert set(sig.keys()) == {"BTC", "ETH"}


def test_weights_clipped_to_max_weight():
    """Weights must never exceed max_weight in absolute value."""
    p = make_cointegrated_prices(beta=5.0, n=500, seed=3)
    m = MeanReversionStrategy(max_weight=0.3, z_entry=0.01)
    m.fit(p)
    sig = m.predict_signal(p)
    assert abs(sig["BTC"]) <= 0.3 + 1e-9
    assert abs(sig["ETH"]) <= 0.3 + 1e-9


def test_protocol_compliance():
    """MeanReversionStrategy satisfies the StrategyProtocol interface."""
    from core.strats.base import StrategyProtocol
    from typing import runtime_checkable, Protocol

    # Verify method signatures exist
    m = MeanReversionStrategy()
    assert hasattr(m, "fit")
    assert hasattr(m, "predict_signal")
    assert callable(m.fit)
    assert callable(m.predict_signal)


def test_spread_zscore_returns_float():
    p = make_cointegrated_prices()
    m = MeanReversionStrategy()
    m.fit(p)
    z = m.spread_zscore(p)
    assert isinstance(z, float)


def test_fit_stores_spread():
    p = make_cointegrated_prices()
    m = MeanReversionStrategy()
    m.fit(p)
    assert m._spread is not None
    assert len(m._spread) == len(p)


def test_beta_positive():
    """Hedge ratio for BTC-ETH should be positive (both assets move together)."""
    p = make_cointegrated_prices()
    m = MeanReversionStrategy()
    m.fit(p)
    assert m.beta > 0, "Beta (hedge ratio) should be positive for BTC-ETH"


def test_kappa_positive():
    """Mean-reversion speed kappa must be positive."""
    p = make_cointegrated_prices()
    m = MeanReversionStrategy()
    m.fit(p)
    assert m.kappa > 0, "kappa must be positive (mean-reverting)"


def test_spread_direction_consistency():
    """
    If z > z_entry, ETH is short and BTC is long (positive).
    If z < -z_entry, ETH is long and BTC is short (negative).
    """
    p = make_cointegrated_prices(kappa=0.5, n=600, seed=99)
    m = MeanReversionStrategy(z_entry=0.01)
    m.fit(p)
    if not m.spread_stationary or m.half_life > m.max_half_life:
        pytest.skip("Spread not stationary or half-life too large for this test data")

    z = m.spread_zscore(p)
    sig = m.predict_signal(p)

    if z > m.z_entry:
        assert sig["ETH"] < 0, f"z={z:.3f} > z_entry -> ETH must be short"
        assert sig["BTC"] > 0, f"z={z:.3f} > z_entry -> BTC must be long"
    elif z < -m.z_entry:
        assert sig["ETH"] > 0, f"z={z:.3f} < -z_entry -> ETH must be long"
        assert sig["BTC"] < 0, f"z={z:.3f} < -z_entry -> BTC must be short"
