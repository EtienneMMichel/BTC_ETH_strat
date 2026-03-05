"""Markowitz portfolio optimisation solvers for 2-asset (BTC/ETH) portfolios.

All public functions take ``mu`` (expected return vector, shape ``(2,)``) and/or
``sigma_matrix`` (2×2 covariance matrix) as inputs and return a weight vector
``w`` of shape ``(2,)``.  Normalisation to ``sum(|w|) ≤ 1`` is handled by
``MarkowitzStrategy``, not here.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _check_inputs(
    mu: np.ndarray | None,
    sigma_matrix: np.ndarray,
    eps: float = 1e-8,
) -> tuple[np.ndarray | None, np.ndarray]:
    """Validate inputs and regularise Sigma if near-singular."""
    sigma_matrix = np.asarray(sigma_matrix, dtype=float)
    if sigma_matrix.shape != (2, 2):
        raise ValueError(f"sigma_matrix must be 2×2, got {sigma_matrix.shape}")
    if not np.all(np.isfinite(sigma_matrix)):
        raise ValueError("sigma_matrix contains non-finite values")

    # Regularise near-singular matrix
    eigvals = np.linalg.eigvalsh(sigma_matrix)
    min_eig = eigvals.min()
    if min_eig < eps:
        sigma_matrix = sigma_matrix + (eps - min_eig) * np.eye(2)

    if mu is not None:
        mu = np.asarray(mu, dtype=float)
        if mu.shape != (2,):
            raise ValueError(f"mu must have shape (2,), got {mu.shape}")
        if not np.all(np.isfinite(mu)):
            # Replace NaN/Inf with zeros
            mu = np.where(np.isfinite(mu), mu, 0.0)

    return mu, sigma_matrix


def _portfolio_vol(w: np.ndarray, sigma_matrix: np.ndarray) -> float:
    """Return portfolio standard deviation sqrt(w'Σw)."""
    var = float(w @ sigma_matrix @ w)
    return float(np.sqrt(max(var, 1e-16)))


def _build_constraints(
    eq_sum_to_one: bool,
    long_only: bool,
    max_w: float,
    min_w: float,
) -> tuple[list[dict], list[tuple]]:
    """Return (constraints, bounds) for scipy.optimize.minimize."""
    constraints: list[dict] = []
    if eq_sum_to_one:
        constraints.append({"type": "eq", "fun": lambda w: w.sum() - 1.0})

    lo = 0.0 if long_only else min_w
    bounds = [(lo, max_w)] * 2
    return constraints, bounds


def _slsqp_solve(
    objective_fn,  # callable(w) -> float to MINIMISE
    n: int = 2,
    eq_sum_to_one: bool = True,
    long_only: bool = False,
    max_w: float = 1.0,
    min_w: float = -1.0,
    n_restarts: int = 3,
) -> np.ndarray:
    """Minimise ``objective_fn`` via SLSQP with multiple random restarts."""
    constraints, bounds = _build_constraints(eq_sum_to_one, long_only, max_w, min_w)

    best_w: np.ndarray = np.full(n, 1.0 / n)
    best_val = np.inf
    rng = np.random.default_rng(42)

    # Always include uniform start plus random restarts
    starts: list[np.ndarray] = [np.full(n, 1.0 / n)]
    for _ in range(n_restarts - 1):
        raw = rng.dirichlet(np.ones(n)) if long_only else rng.uniform(-1, 1, n)
        starts.append(raw)

    for x0 in starts:
        result = minimize(
            objective_fn,
            x0,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"ftol": 1e-12, "maxiter": 500},
        )
        if result.success and result.fun < best_val:
            best_val = result.fun
            best_w = result.x

    return best_w


# ---------------------------------------------------------------------------
# Public solvers
# ---------------------------------------------------------------------------

def max_sharpe_weights(
    mu: np.ndarray,
    sigma_matrix: np.ndarray,
    risk_free: float = 0.0,
    long_only: bool = False,
    max_w: float = 1.0,
    min_w: float = -1.0,
) -> np.ndarray:
    """Maximum Sharpe ratio portfolio weights.

    Analytical solution (unconstrained):
        z = Σ⁻¹ (μ − rf);  w = z / sum(z)
    Falls back to ``min_variance_weights`` if ``||μ − rf|| < 1e-8``.
    Uses SLSQP when ``long_only=True`` or constraints are binding.

    Parameters
    ----------
    mu:
        Expected return vector, shape ``(2,)``.
    sigma_matrix:
        2×2 covariance matrix.
    risk_free:
        Daily risk-free rate.
    long_only:
        Restrict weights to ``[0, max_w]``.
    max_w / min_w:
        Weight bounds (ignored when ``long_only=True`` for the lower bound).
    """
    mu, sigma_matrix = _check_inputs(mu, sigma_matrix)
    excess = mu - risk_free

    # Degenerate case: no excess return signal
    if np.linalg.norm(excess) < 1e-8:
        return min_variance_weights(sigma_matrix, long_only=long_only, max_w=max_w, min_w=min_w)

    # Analytical (unconstrained)
    if not long_only and max_w >= 1.0 and min_w <= -1.0:
        sigma_inv = np.linalg.inv(sigma_matrix)
        z = sigma_inv @ excess
        s = z.sum()
        if abs(s) < 1e-12:
            return min_variance_weights(sigma_matrix, long_only=long_only, max_w=max_w, min_w=min_w)
        return z / s

    # Constrained via SLSQP: maximise Sharpe <=> minimise negative Sharpe
    def neg_sharpe(w: np.ndarray) -> float:
        port_ret = float(w @ excess)
        port_vol = _portfolio_vol(w, sigma_matrix)
        return -port_ret / (port_vol + 1e-12)

    return _slsqp_solve(neg_sharpe, long_only=long_only, max_w=max_w, min_w=min_w)


def min_variance_weights(
    sigma_matrix: np.ndarray,
    long_only: bool = False,
    max_w: float = 1.0,
    min_w: float = -1.0,
) -> np.ndarray:
    """Minimum variance portfolio weights.

    Analytical solution (2-asset, unconstrained):
        w_BTC = (var_ETH − cov) / (var_BTC + var_ETH − 2·cov)
        w_ETH = 1 − w_BTC

    Uses SLSQP when constraints are binding.
    """
    _, sigma_matrix = _check_inputs(None, sigma_matrix)
    var_btc = sigma_matrix[0, 0]
    var_eth = sigma_matrix[1, 1]
    cov = sigma_matrix[0, 1]

    # Analytical (unconstrained)
    if not long_only and max_w >= 1.0 and min_w <= -1.0:
        denom = var_btc + var_eth - 2 * cov
        if abs(denom) < 1e-12:
            return np.array([0.5, 0.5])
        w_btc = (var_eth - cov) / denom
        return np.array([w_btc, 1.0 - w_btc])

    # Constrained
    def port_var(w: np.ndarray) -> float:
        return float(w @ sigma_matrix @ w)

    return _slsqp_solve(port_var, long_only=long_only, max_w=max_w, min_w=min_w)


def mean_variance_weights(
    mu: np.ndarray,
    sigma_matrix: np.ndarray,
    gamma: float = 1.0,
    long_only: bool = False,
    max_w: float = 1.0,
    min_w: float = -1.0,
) -> np.ndarray:
    """Mean-variance utility maximisation weights.

    Analytical solution (unconstrained):
        w = (1/γ) Σ⁻¹ μ

    Uses SLSQP when constraints are binding.

    Parameters
    ----------
    gamma:
        Risk-aversion coefficient (default 1.0).  Higher values reduce
        position size.
    """
    mu, sigma_matrix = _check_inputs(mu, sigma_matrix)
    gamma = max(gamma, 1e-8)

    # Analytical (unconstrained, no sum-to-one constraint)
    if not long_only and max_w >= 1.0 and min_w <= -1.0:
        sigma_inv = np.linalg.inv(sigma_matrix)
        return (1.0 / gamma) * (sigma_inv @ mu)

    # Constrained (maximise mean-variance utility = w'μ - γ/2 w'Σw)
    def neg_utility(w: np.ndarray) -> float:
        return -(float(w @ mu) - 0.5 * gamma * float(w @ sigma_matrix @ w))

    return _slsqp_solve(
        neg_utility,
        eq_sum_to_one=False,
        long_only=long_only,
        max_w=max_w,
        min_w=min_w,
    )


def max_diversification_weights(
    sigma_matrix: np.ndarray,
    long_only: bool = False,
    max_w: float = 1.0,
    min_w: float = -1.0,
) -> np.ndarray:
    """Maximum diversification ratio portfolio weights.

    DR(w) = (w'σ_vec) / sqrt(w'Σw)

    Always solved via SLSQP (non-convex objective).

    Parameters
    ----------
    sigma_matrix:
        2×2 covariance matrix.
    """
    _, sigma_matrix = _check_inputs(None, sigma_matrix)
    sigma_vec = np.sqrt(np.diag(sigma_matrix))

    def neg_dr(w: np.ndarray) -> float:
        weighted_vols = float(w @ sigma_vec)
        port_vol = _portfolio_vol(w, sigma_matrix)
        return -weighted_vols / (port_vol + 1e-12)

    lo = 0.0 if long_only else min_w
    # Max-diversification makes sense only for long-only; enforce at least [0,1]
    effective_long_only = long_only or (lo >= 0)
    return _slsqp_solve(
        neg_dr,
        long_only=effective_long_only,
        max_w=max_w,
        min_w=min_w if not effective_long_only else 0.0,
    )
