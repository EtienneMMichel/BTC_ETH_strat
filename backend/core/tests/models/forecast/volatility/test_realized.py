import numpy as np
import pandas as pd
import pytest
from core.models.forecast.volatility.realized import rogers_satchell, yang_zhang


def make_ohlcv(n=100, seed=42):
    rng = np.random.default_rng(seed)
    close = 100 * np.exp(np.cumsum(rng.standard_normal(n) * 0.01))
    high = close * (1 + np.abs(rng.standard_normal(n)) * 0.005)
    low = close * (1 - np.abs(rng.standard_normal(n)) * 0.005)
    open_ = close * (1 + rng.standard_normal(n) * 0.003)
    # Ensure OHLC consistency
    high = np.maximum(high, np.maximum(open_, close))
    low = np.minimum(low, np.minimum(open_, close))
    idx = pd.date_range("2020-01-01", periods=n, freq="D")
    return pd.DataFrame({"open": open_, "high": high, "low": low, "close": close}, index=idx)


def test_rs_nonnegative():
    ohlcv = make_ohlcv()
    result = rogers_satchell(ohlcv)
    assert (result >= 0).all()


def test_rs_length():
    ohlcv = make_ohlcv(100)
    result = rogers_satchell(ohlcv)
    assert len(result) == 100


def test_yz_nonnegative():
    ohlcv = make_ohlcv()
    result = yang_zhang(ohlcv)
    # Drop NaN at start of rolling window
    assert (result.dropna() >= 0).all()


def test_annualize():
    ohlcv = make_ohlcv()
    r1 = rogers_satchell(ohlcv, annualize=False)
    r2 = rogers_satchell(ohlcv, annualize=True)
    ratio = (r2 / r1).dropna()
    assert np.allclose(ratio, np.sqrt(252), rtol=1e-6)
