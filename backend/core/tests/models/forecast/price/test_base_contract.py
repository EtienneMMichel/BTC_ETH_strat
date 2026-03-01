import numpy as np
import pandas as pd
import pytest
from core.models.forecast.price.momentum import TSMOMModel, MomentumModel
from core.models.forecast.price.trend import EMACrossover, HPFilter, KalmanTrend

MODELS = [
    (TSMOMModel, {}),
    (MomentumModel, {}),
    (EMACrossover, {}),
    (HPFilter, {}),
    (KalmanTrend, {}),
]


def make_prices(n=300):
    rng = np.random.default_rng(99)
    idx = pd.date_range("2021-01-01", periods=n, freq="D")
    return pd.Series(100 * np.exp(np.cumsum(rng.standard_normal(n) * 0.01)), index=idx)


@pytest.mark.parametrize("ModelClass,kwargs", MODELS)
def test_index_alignment(ModelClass, kwargs):
    p = make_prices()
    m = ModelClass(**kwargs)
    m.fit(p)
    pred = m.predict(p)
    assert pred.index.equals(p.index), f"{ModelClass.__name__}: index mismatch"


@pytest.mark.parametrize("ModelClass,kwargs", MODELS)
def test_signal_finite(ModelClass, kwargs):
    p = make_prices()
    m = ModelClass(**kwargs)
    m.fit(p)
    pred = m.predict(p)
    assert np.all(np.isfinite(pred.dropna().values)), f"{ModelClass.__name__}: non-finite signal"
