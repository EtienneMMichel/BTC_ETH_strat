import numpy as np
import pandas as pd
import pytest
from core.evaluation.reports.compare import comparison_table, diebold_mariano


def make_forecast(n=200, mu=0.02, noise=0.005, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2020-01-01", periods=n, freq="D")
    return pd.Series(np.clip(rng.normal(mu, noise, n), 1e-4, None), index=idx)


def test_comparison_table_shape():
    fc1 = make_forecast(seed=0)
    fc2 = make_forecast(seed=1)
    rv = make_forecast(seed=2)
    table = comparison_table({"ModelA": fc1, "ModelB": fc2}, rv)
    assert table.shape == (2, 3)  # 2 models, 3 default metrics


def test_comparison_table_columns():
    fc1 = make_forecast(seed=0)
    rv = make_forecast(seed=2)
    table = comparison_table({"M": fc1}, rv, metrics=["qlike", "mse"])
    assert set(table.columns) == {"qlike", "mse"}


def test_comparison_table_row_index():
    fc1 = make_forecast(seed=0)
    fc2 = make_forecast(seed=1)
    rv = make_forecast(seed=2)
    table = comparison_table({"Alpha": fc1, "Beta": fc2}, rv)
    assert "Alpha" in table.index
    assert "Beta" in table.index


def test_comparison_table_perfect_forecast_zero_mse():
    rv = make_forecast(seed=2)
    table = comparison_table({"Perfect": rv}, rv, metrics=["mse"])
    assert table.loc["Perfect", "mse"] == pytest.approx(0.0, abs=1e-15)


def test_diebold_mariano_keys():
    fc_a = make_forecast(seed=0)
    fc_b = make_forecast(seed=1)
    rv = make_forecast(seed=2)
    result = diebold_mariano(fc_a, fc_b, rv)
    assert set(result.keys()) == {"dm_stat", "p_value"}


def test_diebold_mariano_identical_forecasts():
    fc = make_forecast(seed=0)
    rv = make_forecast(seed=2)
    result = diebold_mariano(fc, fc, rv)
    # Identical forecasts → loss diff = 0 → DM stat = 0
    assert result["dm_stat"] == pytest.approx(0.0, abs=1e-9)


def test_diebold_mariano_clearly_worse_model():
    # fc_a is much worse (noisy), fc_b is perfect
    rv = make_forecast(seed=2)
    rng = np.random.default_rng(99)
    fc_a = pd.Series(
        np.clip(rng.normal(0.05, 0.02, len(rv)), 1e-4, None), index=rv.index
    )
    result = diebold_mariano(fc_a, rv, rv, loss="mse")
    # fc_a is worse than rv (perfect) → positive DM stat
    assert result["dm_stat"] > 0


def test_diebold_mariano_all_losses():
    fc_a = make_forecast(seed=0)
    fc_b = make_forecast(seed=1)
    rv = make_forecast(seed=2)
    for loss in ["mse", "mae", "qlike"]:
        result = diebold_mariano(fc_a, fc_b, rv, loss=loss)
        assert np.isfinite(result["dm_stat"])


def test_comparison_table_unknown_metric_raises():
    fc = make_forecast()
    rv = make_forecast(seed=1)
    with pytest.raises(ValueError, match="Unknown metric"):
        comparison_table({"M": fc}, rv, metrics=["unknown"])
