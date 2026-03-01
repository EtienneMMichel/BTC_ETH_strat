import numpy as np
import pandas as pd
import pytest
from core.models.co_mov.tail.es_cavar import ESCAViaRModel


def make_bivariate_returns(n=400, seed=5):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2020-01-01", periods=n, freq="D")
    btc = pd.Series(rng.standard_normal(n) * 0.03, index=idx)
    eth = pd.Series(rng.standard_normal(n) * 0.04, index=idx)
    return pd.DataFrame({'BTC': btc, 'ETH': eth})


def test_var_always_negative():
    df = make_bivariate_returns()
    m = ESCAViaRModel(alpha=0.05)
    m.fit(df)
    pred = m.predict(df)
    assert (pred['var'] < 0).all(), "VaR should always be negative (left tail)"


def test_es_leq_var():
    df = make_bivariate_returns()
    m = ESCAViaRModel(alpha=0.05)
    m.fit(df)
    pred = m.predict(df)
    # ES should be more negative than VaR (ES <= VaR in signed terms)
    assert (pred['es'] <= pred['var']).all(), "ES must be <= VaR"


def test_shape():
    df = make_bivariate_returns(300)
    m = ESCAViaRModel()
    m.fit(df)
    pred = m.predict(df)
    assert len(pred) == 300
    assert pred.index.equals(df.index)
