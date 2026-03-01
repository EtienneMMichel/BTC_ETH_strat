import numpy as np
import pandas as pd
import pytest
from core.models.forecast.volatility.garch import GARCHModel, GJRGARCHModel, EGARCHModel


def make_garch_series(n=500, seed=42):
    rng = np.random.default_rng(seed)
    returns = pd.Series(
        rng.standard_normal(n) * 0.02,
        index=pd.date_range("2020-01-01", periods=n, freq="D")
    )
    return returns


def test_garch_predict_positive():
    r = make_garch_series()
    m = GARCHModel()
    m.fit(r)
    pred = m.predict(r)
    assert (pred > 0).all()
    assert len(pred) == len(r)
    assert pred.index.equals(r.index)


def test_garch_aic_finite():
    r = make_garch_series()
    m = GARCHModel()
    m.fit(r)
    assert np.isfinite(m.aic)
    assert np.isfinite(m.bic)


def test_gjr_garch_predict_positive():
    r = make_garch_series()
    m = GJRGARCHModel()
    m.fit(r)
    pred = m.predict(r)
    assert (pred > 0).all()
    assert len(pred) == len(r)


def test_egarch_predict_positive():
    r = make_garch_series()
    m = EGARCHModel()
    m.fit(r)
    pred = m.predict(r)
    assert (pred > 0).all()
    assert len(pred) == len(r)
