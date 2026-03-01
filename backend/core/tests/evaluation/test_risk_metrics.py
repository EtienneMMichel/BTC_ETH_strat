import numpy as np
import pandas as pd
import pytest
from core.evaluation.metrics.risk import kupiec_test, christoffersen_test


def make_violations(T=250, n_violations=13, seed=0):
    """Create a violation series with exactly n_violations out of T."""
    rng = np.random.default_rng(seed)
    v = np.zeros(T, dtype=bool)
    idx = rng.choice(T, n_violations, replace=False)
    v[idx] = True
    return pd.Series(v, index=pd.date_range("2020-01-01", periods=T, freq="D"))


def test_kupiec_violation_rate():
    # 13 violations out of 250 → rate ≈ 0.052
    v = make_violations(250, 13)
    result = kupiec_test(v, alpha=0.05)
    assert result["violation_rate"] == pytest.approx(13/250, rel=1e-9)


def test_kupiec_correct_model_high_pvalue():
    # Exactly 5% violation rate → should NOT reject H0 (high p-value)
    T = 1000
    n = 50  # exactly 5%
    v = make_violations(T, n)
    result = kupiec_test(v, alpha=0.05)
    assert result["p_value"] > 0.05


def test_kupiec_bad_model_low_pvalue():
    # 20% violation rate when expecting 5% → should reject H0
    T = 500
    n = 100  # 20%
    v = make_violations(T, n)
    result = kupiec_test(v, alpha=0.05)
    assert result["p_value"] < 0.05


def test_kupiec_returns_required_keys():
    v = make_violations()
    result = kupiec_test(v, 0.05)
    assert set(result.keys()) == {"lr_stat", "p_value", "violation_rate"}


def test_kupiec_lr_stat_nonnegative():
    v = make_violations()
    result = kupiec_test(v, 0.05)
    assert result["lr_stat"] >= 0


def test_christoffersen_returns_required_keys():
    v = make_violations()
    result = christoffersen_test(v, 0.05)
    assert set(result.keys()) == {"lr_stat", "p_value", "lr_uc", "lr_ind"}


def test_christoffersen_lr_cc_geq_lr_uc():
    v = make_violations()
    result = christoffersen_test(v, 0.05)
    assert result["lr_stat"] >= result["lr_uc"] - 1e-10


def test_christoffersen_clustered_violations():
    # Clustered violations: all at the start → strong independence rejection
    T = 500
    v = np.zeros(T, dtype=bool)
    v[:50] = True   # 50 consecutive violations
    violations = pd.Series(v, index=pd.date_range("2020-01-01", periods=T, freq="D"))
    result = christoffersen_test(violations, alpha=0.05)
    # lr_ind should be large (clustered violations violate independence)
    assert result["lr_ind"] > 5.0
