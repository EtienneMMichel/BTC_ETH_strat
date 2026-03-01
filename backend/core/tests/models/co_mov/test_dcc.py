import numpy as np
import pandas as pd
import pytest
from core.models.co_mov.correlation.dcc import DCCModel


def make_bivariate_returns(n=300, rho=0.7, seed=42):
    rng = np.random.default_rng(seed)
    cov = np.array([[1.0, rho], [rho, 1.0]]) * 0.0004  # daily vol ~2%
    raw = rng.multivariate_normal([0, 0], cov, size=n)
    idx = pd.date_range("2020-01-01", periods=n, freq="D")
    return pd.DataFrame(raw, columns=['BTC', 'ETH'], index=idx)


def test_dcc_correlation_in_bounds():
    df = make_bivariate_returns()
    m = DCCModel()
    m.fit(df)
    pred = m.predict(df)
    assert 'correlation' in pred.columns
    assert (pred['correlation'] > -1).all()
    assert (pred['correlation'] < 1).all()


def test_dcc_shape():
    df = make_bivariate_returns(200)
    m = DCCModel()
    m.fit(df)
    pred = m.predict(df)
    assert len(pred) == 200
    assert pred.index.equals(df.index)


def test_dcc_lower_tail_dep():
    df = make_bivariate_returns()
    m = DCCModel()
    m.fit(df)
    pred = m.predict(df)
    assert 'lower_tail_dep' in pred.columns
    assert (pred['lower_tail_dep'] >= 0).all()
