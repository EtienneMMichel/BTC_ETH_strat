import numpy as np
import pandas as pd
import pytest
from core.models.co_mov.correlation.bekk import BEKKModel


def make_bivariate_returns(n=200, rho=0.6, seed=1):
    rng = np.random.default_rng(seed)
    cov = np.array([[1.0, rho], [rho, 1.0]]) * 0.0004
    raw = rng.multivariate_normal([0, 0], cov, size=n)
    idx = pd.date_range("2020-01-01", periods=n, freq="D")
    return pd.DataFrame(raw, columns=['BTC', 'ETH'], index=idx)


def test_bekk_correlation_in_bounds():
    df = make_bivariate_returns()
    m = BEKKModel()
    m.fit(df)
    pred = m.predict(df)
    assert (pred['correlation'] > -1).all()
    assert (pred['correlation'] < 1).all()


def test_bekk_shape():
    df = make_bivariate_returns(150)
    m = BEKKModel()
    m.fit(df)
    pred = m.predict(df)
    assert len(pred) == 150
    assert pred.index.equals(df.index)


def test_bekk_has_required_columns():
    df = make_bivariate_returns()
    m = BEKKModel()
    m.fit(df)
    pred = m.predict(df)
    assert 'correlation' in pred.columns
    assert 'lower_tail_dep' in pred.columns
    assert 'cov_BTC_ETH' in pred.columns
