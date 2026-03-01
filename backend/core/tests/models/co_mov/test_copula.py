import numpy as np
import pandas as pd
import pytest
from core.models.co_mov.tail.copula import (
    GaussianCopula,
    StudentTCopula,
    ClaytonCopula,
    RollingCopula,
)


def perfect_corr_uniform(n=500, seed=0):
    rng = np.random.default_rng(seed)
    u = rng.uniform(0.01, 0.99, n)
    return u, u  # perfect lower tail dep


def independent_uniform(n=500, seed=0):
    rng = np.random.default_rng(seed)
    u = rng.uniform(0.01, 0.99, n)
    v = rng.uniform(0.01, 0.99, n)
    return u, v


def test_gaussian_no_tail_dep():
    u, v = perfect_corr_uniform()
    cop = GaussianCopula()
    cop.fit(u, v)
    assert cop.lower_tail_dep() == 0.0
    assert cop.upper_tail_dep() == 0.0


def test_clayton_perfect_corr_tail_dep():
    u, v = perfect_corr_uniform()
    cop = ClaytonCopula()
    cop.fit(u, v)
    # lambda_L = 2^(-1/theta); with high theta (high correlation), approaches 1
    assert cop.lower_tail_dep() > 0.5


def test_clayton_independent_near_zero():
    u, v = independent_uniform()
    cop = ClaytonCopula()
    cop.fit(u, v)
    # With independent data, tail dep should be low (not necessarily 0 due to estimation)
    assert cop.lower_tail_dep() < 0.5


def test_student_t_symmetric():
    u, v = perfect_corr_uniform()
    cop = StudentTCopula(nu=5)
    cop.fit(u, v)
    assert abs(cop.lower_tail_dep() - cop.upper_tail_dep()) < 1e-10


def test_rolling_copula_shape():
    rng = np.random.default_rng(7)
    n = 200
    idx = pd.date_range("2020-01-01", periods=n, freq="D")
    u = pd.Series(rng.uniform(0.01, 0.99, n), index=idx)
    v = pd.Series(rng.uniform(0.01, 0.99, n), index=idx)
    rc = RollingCopula(ClaytonCopula, window=50)
    result = rc.fit_predict(u, v)
    assert len(result) == n
    assert 'lower_tail_dep' in result.columns
