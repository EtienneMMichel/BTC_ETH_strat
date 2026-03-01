"""
Tests for RegimeClassifier.

Notes
-----
- ADF rolling is O(n * window), so keep n small (<=400) for test speed.
- beta_window is reduced to 63 in some fixtures to speed up the rolling ADF.
"""

import numpy as np
import pandas as pd
import pytest

from core.strats.orchestrator.regime import RegimeClassifier


# ---------------------------------------------------------------------------
# Price fixtures
# ---------------------------------------------------------------------------


def trending_prices(n: int = 400, seed: int = 0) -> pd.DataFrame:
    """Persistent uptrend, low vol."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2020-01-01", periods=n, freq="D")
    btc = np.exp(np.cumsum(rng.standard_normal(n) * 0.008 + 0.002)) * 10000
    eth = np.exp(np.cumsum(rng.standard_normal(n) * 0.010 + 0.0015)) * 500
    return pd.DataFrame({"BTC": btc, "ETH": eth}, index=idx)


def make_prices(n: int = 300, seed: int = 1) -> pd.DataFrame:
    """Generic random-walk prices."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2020-01-01", periods=n, freq="D")
    btc = np.exp(np.cumsum(rng.standard_normal(n) * 0.02)) * 10000
    eth = np.exp(np.cumsum(rng.standard_normal(n) * 0.025)) * 500
    return pd.DataFrame({"BTC": btc, "ETH": eth}, index=idx)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_classify_length():
    """Output length must match input price length."""
    p = make_prices()
    clf = RegimeClassifier(beta_window=63)
    regimes = clf.classify(p)
    assert len(regimes) == len(p)


def test_classify_valid_labels():
    """All regime values must be in the allowed set."""
    p = make_prices()
    clf = RegimeClassifier(beta_window=63)
    regimes = clf.classify(p)
    assert set(regimes.unique()).issubset({"momentum", "mean_reversion", "cash"})


def test_hysteresis_min_holding():
    """Regime must not switch faster than min_holding_bars."""
    p = make_prices(400)
    clf = RegimeClassifier(min_holding_bars=10, beta_window=63)
    regimes = clf.classify(p)
    transitions = (regimes != regimes.shift(1)).sum()
    # With 400 bars and min_holding=10, at most 40 transitions (+2 slack for
    # edge effects).
    assert transitions <= 400 // clf.min_holding_bars + 2


def test_classify_index():
    """Output index must be identical to the input price index."""
    p = make_prices()
    clf = RegimeClassifier(beta_window=63)
    regimes = clf.classify(p)
    assert regimes.index.equals(p.index)
