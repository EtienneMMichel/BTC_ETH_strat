import numpy as np
import pandas as pd
import pytest
from core.models.covariance.bekk_cov import BEKKCovModel


def make_returns(n=200, seed=3):
    rng = np.random.default_rng(seed)
    cov = np.array([[0.0004, 0.00020], [0.00020, 0.0005]])
    raw = rng.multivariate_normal([0, 0], cov, size=n)
    idx = pd.date_range("2020-01-01", periods=n, freq="D", tz="UTC")
    return pd.DataFrame(raw, columns=["BTC", "ETH"], index=idx)


def test_bekk_cov_columns():
    df = make_returns()
    m = BEKKCovModel()
    m.fit(df)
    pred = m.predict(df)
    assert set(pred.columns) == {"var_BTC", "cov_BTC_ETH", "var_ETH"}


def test_bekk_cov_index():
    df = make_returns()
    m = BEKKCovModel()
    m.fit(df)
    pred = m.predict(df)
    assert pred.index.equals(df.index)


def test_bekk_cov_variances_positive():
    df = make_returns()
    m = BEKKCovModel()
    m.fit(df)
    pred = m.predict(df)
    assert (pred["var_BTC"] > 0).all()
    assert (pred["var_ETH"] > 0).all()


def test_bekk_cov_shape():
    df = make_returns(150)
    m = BEKKCovModel()
    m.fit(df)
    pred = m.predict(df)
    assert len(pred) == 150


def test_bekk_cov_not_fitted_raises():
    m = BEKKCovModel()
    df = make_returns(50)
    with pytest.raises(RuntimeError, match="not been fitted"):
        m.predict(df)


def test_bekk_cov_parameters_copied():
    df = make_returns()
    m = BEKKCovModel()
    m.fit(df)
    assert m._C is not None
    assert m._A is not None
    assert m._B is not None
    assert m._H0 is not None


def test_bekk_cov_persistence():
    """BEKK with high persistence (B=0.85I) should produce smooth variance series."""
    df = make_returns(200)
    m = BEKKCovModel()
    m.fit(df)
    pred = m.predict(df)
    # Check that day-over-day changes in var_BTC are small relative to the level
    rel_change = (pred["var_BTC"].diff().abs() / pred["var_BTC"]).dropna()
    assert rel_change.median() < 0.5
