import numpy as np
import pytest
from core.strats.markowitz.optimizer import (
    max_sharpe_weights,
    min_variance_weights,
    mean_variance_weights,
    max_diversification_weights,
    _portfolio_vol,
    _check_inputs,
)


# ---- helpers ---------------------------------------------------------------

def simple_sigma(var_btc=0.0004, var_eth=0.0006, cov=0.0002):
    return np.array([[var_btc, cov], [cov, var_eth]])


def simple_mu(btc=0.001, eth=0.0008):
    return np.array([btc, eth])


# ---- _check_inputs ---------------------------------------------------------

def test_check_inputs_bad_shape():
    with pytest.raises(ValueError, match="2×2"):
        _check_inputs(None, np.eye(3))


def test_check_inputs_regularises_near_singular():
    sigma = np.array([[1e-12, 0.0], [0.0, 1e-12]])
    _, sigma_reg = _check_inputs(None, sigma)
    eigvals = np.linalg.eigvalsh(sigma_reg)
    assert eigvals.min() >= 1e-8


def test_check_inputs_nan_mu_replaced():
    mu = np.array([np.nan, 0.001])
    mu_out, _ = _check_inputs(mu, simple_sigma())
    assert np.isfinite(mu_out).all()
    assert mu_out[0] == 0.0


# ---- min_variance_weights --------------------------------------------------

def test_min_var_unconstrained_sum_to_one():
    w = min_variance_weights(simple_sigma())
    assert abs(w.sum() - 1.0) < 1e-8


def test_min_var_shape():
    w = min_variance_weights(simple_sigma())
    assert w.shape == (2,)


def test_min_var_long_only():
    w = min_variance_weights(simple_sigma(), long_only=True)
    assert (w >= -1e-9).all()
    assert abs(w.sum() - 1.0) < 1e-6


def test_min_var_lower_than_equal_weight():
    sigma = simple_sigma()
    w_mv = min_variance_weights(sigma)
    w_eq = np.array([0.5, 0.5])
    assert _portfolio_vol(w_mv, sigma) <= _portfolio_vol(w_eq, sigma) + 1e-8


def test_min_var_uncorrelated_equal_var():
    """Uncorrelated equal-variance assets → 50/50 split."""
    sigma = np.array([[0.01, 0.0], [0.0, 0.01]])
    w = min_variance_weights(sigma)
    assert abs(w[0] - 0.5) < 1e-6


# ---- max_sharpe_weights ----------------------------------------------------

def test_max_sharpe_shape():
    w = max_sharpe_weights(simple_mu(), simple_sigma())
    assert w.shape == (2,)


def test_max_sharpe_sum_to_one():
    w = max_sharpe_weights(simple_mu(), simple_sigma())
    assert abs(w.sum() - 1.0) < 1e-6


def test_max_sharpe_zero_mu_falls_back():
    """Near-zero mu should fall back to min_variance."""
    mu = np.array([1e-10, 1e-10])
    sigma = simple_sigma()
    w_sharpe = max_sharpe_weights(mu, sigma)
    w_mv = min_variance_weights(sigma)
    assert abs(w_sharpe[0] - w_mv[0]) < 1e-5


def test_max_sharpe_long_only():
    w = max_sharpe_weights(simple_mu(), simple_sigma(), long_only=True)
    assert (w >= -1e-9).all()


def test_max_sharpe_higher_sharpe_than_equal_weight():
    sigma = simple_sigma()
    mu = simple_mu()
    w_opt = max_sharpe_weights(mu, sigma)
    w_eq = np.array([0.5, 0.5])
    sharpe_opt = float(w_opt @ mu) / _portfolio_vol(w_opt, sigma)
    sharpe_eq = float(w_eq @ mu) / _portfolio_vol(w_eq, sigma)
    assert sharpe_opt >= sharpe_eq - 1e-6


# ---- mean_variance_weights -------------------------------------------------

def test_mean_variance_shape():
    w = mean_variance_weights(simple_mu(), simple_sigma())
    assert w.shape == (2,)


def test_mean_variance_long_only():
    w = mean_variance_weights(simple_mu(), simple_sigma(), long_only=True)
    assert (w >= -1e-9).all()


def test_mean_variance_high_gamma_small_weights():
    """High risk aversion → smaller weights overall."""
    mu = simple_mu()
    sigma = simple_sigma()
    w_low = mean_variance_weights(mu, sigma, gamma=0.5)
    w_high = mean_variance_weights(mu, sigma, gamma=10.0)
    assert np.abs(w_low).sum() >= np.abs(w_high).sum() - 1e-6


# ---- max_diversification_weights -------------------------------------------

def test_max_div_long_only_by_default():
    """Max diversification enforces long positions."""
    w = max_diversification_weights(simple_sigma())
    assert (w >= -1e-9).all()


def test_max_div_shape():
    w = max_diversification_weights(simple_sigma())
    assert w.shape == (2,)


def test_max_div_sum_to_one():
    w = max_diversification_weights(simple_sigma())
    assert abs(w.sum() - 1.0) < 1e-5


def test_max_div_equal_vol_equal_weight():
    """Equal individual vols → equal weights maximise DR."""
    sigma = np.array([[0.01, 0.005], [0.005, 0.01]])
    w = max_diversification_weights(sigma)
    assert abs(w[0] - w[1]) < 1e-4


# ---- _portfolio_vol --------------------------------------------------------

def test_portfolio_vol_known_case():
    sigma = np.array([[0.04, 0.0], [0.0, 0.09]])
    w = np.array([1.0, 0.0])
    assert abs(_portfolio_vol(w, sigma) - 0.2) < 1e-8
