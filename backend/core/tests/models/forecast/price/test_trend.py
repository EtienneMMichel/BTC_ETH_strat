import numpy as np
import pandas as pd
import pytest
from core.models.forecast.price.trend import EMACrossover, HPFilter, KalmanTrend


def make_prices(n=300, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2020-01-01", periods=n, freq="D")
    return pd.Series(100 * np.exp(np.cumsum(rng.standard_normal(n) * 0.01)), index=idx)


def test_ema_crossover_invalid_spans():
    with pytest.raises(ValueError):
        m = EMACrossover(span_fast=200, span_slow=50)
        m.fit(make_prices())


def test_ema_crossover_shape():
    p = make_prices()
    m = EMACrossover()
    m.fit(p)
    pred = m.predict(p)
    assert pred.index.equals(p.index)
    assert pred.dropna().abs().max() <= 1.0 + 1e-6


def test_hp_filter_causal():
    """Signal at t should not depend on future prices."""
    p = make_prices(200)
    m = HPFilter()
    m.fit(p)
    pred_orig = m.predict(p).copy()

    # Perturb future prices (last 50)
    p_perturbed = p.copy()
    p_perturbed.iloc[150:] *= 2.0
    pred_perturbed = m.predict(p_perturbed)

    # First 148 signals should be identical (index 0..149, but shift(1) means 0..148)
    assert np.allclose(
        pred_orig.iloc[:148].dropna().values,
        pred_perturbed.iloc[:148].dropna().values,
        atol=1e-6,
    ), "HP filter is not causal: perturbing future prices changed past signals"


def test_kalman_shape():
    p = make_prices()
    m = KalmanTrend()
    m.fit(p)
    pred = m.predict(p)
    assert len(pred) == len(p)
    assert pred.index.equals(p.index)


def test_kalman_finite():
    p = make_prices()
    m = KalmanTrend()
    m.fit(p)
    pred = m.predict(p)
    assert np.all(np.isfinite(pred.values))
