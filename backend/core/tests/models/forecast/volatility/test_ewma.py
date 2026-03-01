import numpy as np
import pandas as pd
from core.models.forecast.volatility.ewma import EWMAModel


def test_ewma_shape():
    rng = np.random.default_rng(0)
    r = pd.Series(rng.standard_normal(200) * 0.02,
                  index=pd.date_range("2020-01-01", periods=200, freq="D"))
    m = EWMAModel()
    m.fit(r)
    pred = m.predict(r)
    assert len(pred) == len(r)
    assert pred.index.equals(r.index)


def test_ewma_positive():
    rng = np.random.default_rng(1)
    r = pd.Series(rng.standard_normal(200) * 0.02,
                  index=pd.date_range("2020-01-01", periods=200, freq="D"))
    m = EWMAModel()
    m.fit(r)
    pred = m.predict(r)
    assert (pred > 0).all()


def test_ewma_converges_to_true_vol():
    # Constant vol white noise: EWMA should converge close to true sigma
    true_sigma = 0.02
    rng = np.random.default_rng(2)
    r = pd.Series(rng.standard_normal(2000) * true_sigma,
                  index=pd.date_range("2020-01-01", periods=2000, freq="D"))
    m = EWMAModel()
    m.fit(r)
    pred = m.predict(r)
    # Last 500 predictions should be within 20% of true sigma
    assert abs(pred.iloc[-500:].mean() - true_sigma) < true_sigma * 0.2
