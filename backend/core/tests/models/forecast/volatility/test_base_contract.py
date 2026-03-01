import numpy as np
import pandas as pd
import pytest
from core.models.forecast.volatility.garch import GARCHModel, GJRGARCHModel, EGARCHModel
from core.models.forecast.volatility.ewma import EWMAModel

MODELS = [GARCHModel, GJRGARCHModel, EGARCHModel, EWMAModel]


def make_returns(n=300, seed=0):
    rng = np.random.default_rng(seed)
    return pd.Series(
        rng.standard_normal(n) * 0.015,
        index=pd.date_range("2021-01-01", periods=n, freq="D")
    )


@pytest.mark.parametrize("ModelClass", MODELS)
def test_index_alignment(ModelClass):
    r = make_returns()
    m = ModelClass()
    m.fit(r)
    pred = m.predict(r)
    assert pred.index.equals(r.index), f"{ModelClass.__name__}: index mismatch"


@pytest.mark.parametrize("ModelClass", MODELS)
def test_all_positive(ModelClass):
    r = make_returns()
    m = ModelClass()
    m.fit(r)
    pred = m.predict(r)
    assert (pred > 0).all(), f"{ModelClass.__name__}: has non-positive values"
