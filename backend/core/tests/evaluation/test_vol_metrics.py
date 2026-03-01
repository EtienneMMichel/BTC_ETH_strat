import numpy as np
import pandas as pd
import pytest
from core.evaluation.metrics.vol import qlike, mse_vol, mae_vol, mincer_zarnowitz


def make_series(n=200, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2020-01-01", periods=n, freq="D")
    return pd.Series(rng.uniform(0.01, 0.05, n), index=idx)


def test_qlike_perfect_forecast():
    idx = pd.date_range("2020-01-01", periods=100, freq="D")
    s = pd.Series(np.full(100, 0.02), index=idx)
    # Perfect forecast: sigma_hat == sigma → QLIKE = 0
    assert qlike(s, s) == pytest.approx(0.0, abs=1e-10)


def test_qlike_positive():
    fc = make_series(seed=0)
    rv = make_series(seed=1)
    assert qlike(fc, rv) >= 0


def test_qlike_invalid_forecast_returns_nan():
    idx = pd.date_range("2020-01-01", periods=10, freq="D")
    fc = pd.Series(np.full(10, -0.01), index=idx)  # negative forecast
    rv = pd.Series(np.full(10, 0.02), index=idx)
    result = qlike(fc, rv)
    assert np.isnan(result)


def test_mse_perfect_forecast():
    s = make_series()
    assert mse_vol(s, s) == pytest.approx(0.0, abs=1e-15)


def test_mse_positive():
    fc = make_series(seed=0)
    rv = make_series(seed=1)
    assert mse_vol(fc, rv) >= 0


def test_mae_perfect_forecast():
    s = make_series()
    assert mae_vol(s, s) == pytest.approx(0.0, abs=1e-15)


def test_mae_positive():
    fc = make_series(seed=0)
    rv = make_series(seed=1)
    assert mae_vol(fc, rv) >= 0


def test_mincer_zarnowitz_perfect():
    s = make_series()
    result = mincer_zarnowitz(s, s)
    assert result["beta"] == pytest.approx(1.0, abs=1e-6)
    assert result["alpha"] == pytest.approx(0.0, abs=1e-6)
    assert result["r_squared"] == pytest.approx(1.0, abs=1e-6)


def test_mincer_zarnowitz_keys():
    fc = make_series(seed=0)
    rv = make_series(seed=1)
    result = mincer_zarnowitz(fc, rv)
    assert set(result.keys()) == {"alpha", "beta", "r_squared"}


def test_qlike_known_value():
    # sigma_hat = 2, sigma = 1: QLIKE = 1/4 - log(1/4) - 1 = 0.25 + log(4) - 1
    idx = pd.date_range("2020-01-01", periods=1, freq="D")
    fc = pd.Series([2.0], index=idx)
    rv = pd.Series([1.0], index=idx)
    expected = (1.0/4.0) - np.log(1.0/4.0) - 1.0
    assert qlike(fc, rv) == pytest.approx(expected, rel=1e-6)
