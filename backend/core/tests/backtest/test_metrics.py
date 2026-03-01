import numpy as np
import pandas as pd
import pytest
from core.backtest.metrics.perf import (
    sharpe_ratio,
    max_drawdown,
    calmar_ratio,
    historical_var,
    expected_shortfall,
    win_rate,
    compute_all,
)


def flat_returns(n=252, mu=0.001, sigma=0.01, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2022-01-01", periods=n, freq="D")
    return pd.Series(rng.normal(mu, sigma, n), index=idx)


def equity_from_returns(returns):
    return (1 + returns).cumprod()


def test_sharpe_positive_for_positive_mean():
    r = flat_returns(mu=0.002, sigma=0.01)
    sr = sharpe_ratio(r)
    assert sr > 0


def test_sharpe_zero_std():
    r = pd.Series([0.001] * 252)
    # constant positive return — should give very high sharpe
    sr = sharpe_ratio(r)
    assert np.isfinite(sr)


def test_max_drawdown_non_positive():
    r = flat_returns()
    eq = equity_from_returns(r)
    mdd = max_drawdown(eq)
    assert mdd <= 0


def test_max_drawdown_monotone_up():
    eq = pd.Series(np.linspace(1, 2, 100))
    mdd = max_drawdown(eq)
    assert mdd == pytest.approx(0.0, abs=1e-9)


def test_max_drawdown_known():
    # Equity: 1 → 2 → 1 → drawdown = -50%
    eq = pd.Series([1.0, 2.0, 1.0])
    mdd = max_drawdown(eq)
    assert mdd == pytest.approx(-0.5, abs=1e-9)


def test_historical_var_negative():
    r = flat_returns()
    var = historical_var(r)
    assert var < 0


def test_historical_var_rolling_shape():
    r = flat_returns(n=200)
    var_series = historical_var(r, alpha=0.05, window=63)
    assert len(var_series) == 200


def test_expected_shortfall_leq_var():
    r = flat_returns()
    var = historical_var(r)
    es = expected_shortfall(r)
    assert es <= var


def test_win_rate_all_positive():
    tl = pd.DataFrame({"pnl": [1.0, 2.0, 3.0]})
    assert win_rate(tl) == pytest.approx(1.0)


def test_win_rate_all_negative():
    tl = pd.DataFrame({"pnl": [-1.0, -2.0]})
    assert win_rate(tl) == pytest.approx(0.0)


def test_win_rate_empty():
    tl = pd.DataFrame({"pnl": []})
    assert np.isnan(win_rate(tl))


def test_compute_all_keys():
    r = flat_returns()
    eq = equity_from_returns(r)
    tl = pd.DataFrame({"pnl": [1.0, -0.5, 0.3]})
    result = compute_all(eq, tl)
    assert "sharpe_ratio" in result
    assert "max_drawdown" in result
    assert "historical_var_5pct" in result
    assert "expected_shortfall_5pct" in result
