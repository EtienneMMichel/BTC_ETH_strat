import numpy as np
import pandas as pd
import pytest
from core.models.covariance.rolling import RollingCovModel


def make_returns(n=120, seed=0):
    rng = np.random.default_rng(seed)
    cov = np.array([[0.0004, 0.0002], [0.0002, 0.0006]])
    raw = rng.multivariate_normal([0, 0], cov, size=n)
    idx = pd.date_range("2020-01-01", periods=n, freq="D", tz="UTC")
    return pd.DataFrame(raw, columns=["BTC", "ETH"], index=idx)


def test_rolling_cov_columns():
    df = make_returns()
    m = RollingCovModel()
    m.fit(df)
    pred = m.predict(df)
    assert set(pred.columns) == {"var_BTC", "cov_BTC_ETH", "var_ETH"}


def test_rolling_cov_index():
    df = make_returns()
    m = RollingCovModel()
    m.fit(df)
    pred = m.predict(df)
    assert pred.index.equals(df.index)


def test_rolling_cov_variances_positive():
    df = make_returns()
    m = RollingCovModel()
    m.fit(df)
    pred = m.predict(df).dropna()
    assert (pred["var_BTC"] > 0).all()
    assert (pred["var_ETH"] > 0).all()


def test_rolling_cov_no_lookahead():
    """Perturbing the last row must not change any earlier prediction."""
    df = make_returns(100)
    m = RollingCovModel()
    m.fit(df)
    pred_orig = m.predict(df).iloc[:-1].copy()

    df2 = df.copy()
    df2.iloc[-1] *= 10  # large perturbation on final bar
    pred_perturbed = m.predict(df2).iloc[:-1]

    pd.testing.assert_frame_equal(pred_orig, pred_perturbed)


def test_rolling_cov_window_default():
    m = RollingCovModel()
    assert m.window == 63


def test_rolling_cov_short_series():
    """Series shorter than window should still return non-NaN via expanding fallback."""
    df = make_returns(10)
    m = RollingCovModel(window=63, min_periods=2)
    m.fit(df)
    pred = m.predict(df)
    # First row is NaN (shift(1) of first expanding estimate is NaN at position 0)
    # but from row 2 onward (after shift) there should be values
    assert pred.dropna().shape[0] > 0


def test_rolling_cov_custom_window():
    df = make_returns(200)
    m = RollingCovModel(window=21, min_periods=5)
    m.fit(df)
    pred = m.predict(df)
    assert len(pred) == 200
