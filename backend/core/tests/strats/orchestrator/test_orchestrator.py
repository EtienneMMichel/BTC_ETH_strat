"""
Tests for OrchestratorStrategy.

Mock sub-strategies are used so that the tests are independent of whether
MomentumStrategy and MeanReversionStrategy have been implemented yet.
"""

import numpy as np
import pandas as pd
import pytest

from core.strats.orchestrator.regime import OrchestratorStrategy, RegimeClassifier


# ---------------------------------------------------------------------------
# Mock sub-strategies (duck-typed StrategyProtocol)
# ---------------------------------------------------------------------------


class MockMomentum:
    def fit(self, prices: pd.DataFrame) -> None:
        pass

    def predict_signal(self, prices: pd.DataFrame) -> dict:
        return {"BTC": 0.4, "ETH": 0.3}


class MockMeanReversion:
    def fit(self, prices: pd.DataFrame) -> None:
        pass

    def predict_signal(self, prices: pd.DataFrame) -> dict:
        return {"BTC": -0.2, "ETH": 0.2}


# ---------------------------------------------------------------------------
# Price fixture
# ---------------------------------------------------------------------------


def make_prices(n: int = 300, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2020-01-01", periods=n, freq="D")
    btc = np.exp(np.cumsum(rng.standard_normal(n) * 0.02)) * 10000
    eth = np.exp(np.cumsum(rng.standard_normal(n) * 0.025)) * 500
    return pd.DataFrame({"BTC": btc, "ETH": eth}, index=idx)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_predict_signal_keys():
    """predict_signal must always return a dict with keys 'BTC' and 'ETH'."""
    p = make_prices()
    orch = OrchestratorStrategy(MockMomentum(), MockMeanReversion())
    orch.fit(p)
    sig = orch.predict_signal(p)
    assert set(sig.keys()) == {"BTC", "ETH"}


def test_weights_finite():
    """All returned weights must be finite numbers."""
    p = make_prices()
    orch = OrchestratorStrategy(MockMomentum(), MockMeanReversion())
    orch.fit(p)
    sig = orch.predict_signal(p)
    assert all(np.isfinite(v) for v in sig.values())


def test_current_regime_valid():
    """current_regime must return one of the three valid regime labels."""
    p = make_prices()
    orch = OrchestratorStrategy(MockMomentum(), MockMeanReversion())
    orch.fit(p)
    regime = orch.current_regime(p)
    assert regime in {"momentum", "mean_reversion", "cash"}


def test_cash_regime_zero_weights():
    """
    Force the classifier into the 'cash' regime by setting vol_threshold_pct=0.0
    and adf_pvalue=0.0.

    With vol_threshold_pct=0.0:
        avg_vol.expanding().quantile(0.0) == min(avg_vol)
        avg_vol < min(avg_vol) is always False  → is_low_vol always False
    With adf_pvalue=0.0:
        ADF p-value is never < 0.0             → is_stationary always False

    Both conditions False → every bar falls into the 'else: cash' branch,
    so predict_signal must return {"BTC": 0.0, "ETH": 0.0}.
    """
    p = make_prices()
    clf = RegimeClassifier(
        vol_threshold_pct=0.0,
        adf_pvalue=0.0,
    )
    orch = OrchestratorStrategy(MockMomentum(), MockMeanReversion(), clf)
    orch.fit(p)
    sig = orch.predict_signal(p)
    assert sig["BTC"] == 0.0 and sig["ETH"] == 0.0
