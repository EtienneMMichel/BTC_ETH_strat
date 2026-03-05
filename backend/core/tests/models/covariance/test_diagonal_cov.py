import numpy as np
import pandas as pd
import pytest
from core.models.covariance.diagonal import DiagonalCovModel
from core.models.forecast.volatility.ewma import EWMAModel
from core.models.co_mov.correlation.dcc import DCCModel


def make_returns(n=200, seed=7):
    rng = np.random.default_rng(seed)
    cov = np.array([[0.0004, 0.00024], [0.00024, 0.0006]])
    raw = rng.multivariate_normal([0, 0], cov, size=n)
    idx = pd.date_range("2020-01-01", periods=n, freq="D", tz="UTC")
    return pd.DataFrame(raw, columns=["BTC", "ETH"], index=idx)


def test_diagonal_cov_columns():
    df = make_returns()
    m = DiagonalCovModel()
    m.fit(df)
    pred = m.predict(df)
    assert set(pred.columns) == {"var_BTC", "cov_BTC_ETH", "var_ETH"}


def test_diagonal_cov_index():
    df = make_returns()
    m = DiagonalCovModel()
    m.fit(df)
    pred = m.predict(df)
    assert pred.index.equals(df.index)


def test_diagonal_cov_variances_positive():
    df = make_returns()
    m = DiagonalCovModel()
    m.fit(df)
    pred = m.predict(df)
    assert (pred["var_BTC"] > 0).all()
    assert (pred["var_ETH"] > 0).all()


def test_diagonal_cov_default_models():
    m = DiagonalCovModel()
    assert isinstance(m._vol_btc, EWMAModel)
    assert isinstance(m._vol_eth, EWMAModel)
    assert isinstance(m._corr, DCCModel)


def test_diagonal_cov_custom_vol_model():
    """Custom vol model should be wired in correctly."""
    custom_btc = EWMAModel(lam=0.97)
    m = DiagonalCovModel(vol_model_btc=custom_btc)
    assert m._vol_btc is custom_btc


def test_diagonal_cov_shape():
    df = make_returns(150)
    m = DiagonalCovModel()
    m.fit(df)
    pred = m.predict(df)
    assert len(pred) == 150


def test_diagonal_cov_correlation_reflects_dcc():
    """cov_BTC_ETH should have same sign as DCC correlation (positive rho → positive cov)."""
    df = make_returns()
    m = DiagonalCovModel()
    m.fit(df)
    pred = m.predict(df)
    # With positive true correlation, most cov values should be positive
    assert (pred["cov_BTC_ETH"] > 0).mean() > 0.5
