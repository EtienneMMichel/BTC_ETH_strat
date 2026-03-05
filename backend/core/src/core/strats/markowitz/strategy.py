"""Markowitz mean-variance portfolio strategy for BTC/ETH."""
from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd

from core.models.covariance.base import CovarianceModel
from core.models.covariance.diagonal import DiagonalCovModel
from core.models.expected_returns.base import ExpectedReturnsModel
from core.models.expected_returns.signal import SignalExpectedReturns
from core.strats.markowitz.optimizer import (
    max_sharpe_weights,
    min_variance_weights,
    mean_variance_weights,
    max_diversification_weights,
)


class MarkowitzStrategy:
    """Markowitz mean-variance portfolio strategy.

    Implements ``StrategyProtocol`` from ``core.strats.base``.

    Parameters
    ----------
    cov_model:
        Time-varying covariance model (default ``DiagonalCovModel()``).
    expected_returns:
        Daily expected-return estimator (default ``SignalExpectedReturns()``).
    objective:
        Portfolio objective: ``'max_sharpe'``, ``'min_variance'``,
        ``'mean_variance'``, or ``'max_diversification'``.
    gamma:
        Risk-aversion coefficient for the ``'mean_variance'`` objective.
    long_only:
        Restrict weights to ``[0, max_weight]``.
    max_weight:
        Upper weight bound per asset (default 1.0).
    min_weight:
        Lower weight bound per asset (default -1.0); ignored when ``long_only``.
    risk_free_rate:
        Annualised risk-free rate (default 0.0); divided by 252 internally.
    target_vol:
        If set, rescale weights so that portfolio vol equals ``target_vol``
        (annualised daily vol × √252).  Applied before clipping.
    min_history:
        Minimum number of price observations required to produce a signal;
        returns ``{'BTC': 0.0, 'ETH': 0.0}`` for shorter series.
    """

    def __init__(
        self,
        cov_model: CovarianceModel | None = None,
        expected_returns: ExpectedReturnsModel | None = None,
        objective: Literal[
            "max_sharpe", "min_variance", "mean_variance", "max_diversification"
        ] = "max_sharpe",
        gamma: float = 1.0,
        long_only: bool = False,
        max_weight: float = 1.0,
        min_weight: float = -1.0,
        risk_free_rate: float = 0.0,
        target_vol: float | None = None,
        min_history: int = 252,
    ) -> None:
        self._cov_model: CovarianceModel = cov_model if cov_model is not None else DiagonalCovModel()
        self._er_model: ExpectedReturnsModel = (
            expected_returns if expected_returns is not None else SignalExpectedReturns()
        )
        self.objective = objective
        self.gamma = gamma
        self.long_only = long_only
        self.max_weight = max_weight
        self.min_weight = min_weight
        self.risk_free_rate = risk_free_rate
        self.target_vol = target_vol
        self.min_history = min_history

    # ------------------------------------------------------------------
    def fit(self, prices: pd.DataFrame) -> None:
        """Fit internal covariance and expected-return models.

        Parameters
        ----------
        prices:
            DataFrame with columns ``['BTC', 'ETH']``, UTC-indexed.
        """
        log_returns = np.log(prices).diff().dropna()
        self._cov_model.fit(log_returns)
        self._er_model.fit(prices)

    # ------------------------------------------------------------------
    def predict_signal(self, prices: pd.DataFrame) -> dict[str, float]:
        """Compute Markowitz optimal weights for the next bar.

        Steps
        -----
        1. Guard: return zeros if fewer than ``min_history`` observations.
        2. Compute log-returns and predict covariance components → 2×2 Σ.
        3. Regularise Σ if near-singular.
        4. Predict expected daily returns → μ vector (NaN → 0).
        5. Dispatch to the chosen optimiser.
        6. Optional ``target_vol`` rescaling.
        7. Clip to ``[min_weight, max_weight]`` (or ``[0, max_weight]`` if long-only).
        8. Normalise so ``sum(|w|) ≤ 1``.
        """
        if len(prices) < self.min_history:
            return {"BTC": 0.0, "ETH": 0.0}

        log_returns = np.log(prices).diff()
        # Drop the first NaN row (from diff) so the covariance model receives
        # a DataFrame whose length matches what was fitted in fit().
        log_returns_clean = log_returns.dropna()

        # --- Covariance -------------------------------------------------------
        cov_df = self._cov_model.predict(log_returns_clean)
        last_cov = cov_df.iloc[-1]
        var_btc = float(last_cov["var_BTC"])
        var_eth = float(last_cov["var_ETH"])
        cov_btc_eth = float(last_cov["cov_BTC_ETH"])

        sigma_matrix = np.array([[var_btc, cov_btc_eth], [cov_btc_eth, var_eth]])

        # Regularise
        eigvals = np.linalg.eigvalsh(sigma_matrix)
        min_eig = eigvals.min()
        if min_eig < 1e-8:
            sigma_matrix += (1e-8 - min_eig) * np.eye(2)

        # --- Expected returns --------------------------------------------------
        mu_df = self._er_model.predict(prices)
        last_mu = mu_df.iloc[-1]
        mu = np.array([
            float(last_mu["BTC"]) if np.isfinite(last_mu["BTC"]) else 0.0,
            float(last_mu["ETH"]) if np.isfinite(last_mu["ETH"]) else 0.0,
        ])

        rf_daily = self.risk_free_rate / 252.0

        # --- Optimise ----------------------------------------------------------
        if self.objective == "max_sharpe":
            w = max_sharpe_weights(
                mu, sigma_matrix,
                risk_free=rf_daily,
                long_only=self.long_only,
                max_w=self.max_weight,
                min_w=self.min_weight,
            )
        elif self.objective == "min_variance":
            w = min_variance_weights(
                sigma_matrix,
                long_only=self.long_only,
                max_w=self.max_weight,
                min_w=self.min_weight,
            )
        elif self.objective == "mean_variance":
            w = mean_variance_weights(
                mu, sigma_matrix,
                gamma=self.gamma,
                long_only=self.long_only,
                max_w=self.max_weight,
                min_w=self.min_weight,
            )
        elif self.objective == "max_diversification":
            w = max_diversification_weights(
                sigma_matrix,
                long_only=self.long_only,
                max_w=self.max_weight,
                min_w=self.min_weight,
            )
        else:
            raise ValueError(f"Unknown objective: {self.objective!r}")

        # --- target_vol rescaling ---------------------------------------------
        if self.target_vol is not None:
            port_var = float(w @ sigma_matrix @ w)
            port_vol_annual = np.sqrt(max(port_var, 1e-16)) * np.sqrt(252)
            if port_vol_annual > 1e-8:
                w = w * (self.target_vol / port_vol_annual)

        # --- Clip --------------------------------------------------------------
        lo = 0.0 if self.long_only else self.min_weight
        w = np.clip(w, lo, self.max_weight)

        # --- Normalise: sum(|w|) <= 1 -----------------------------------------
        total = float(np.abs(w).sum())
        if total > 1.0:
            w = w / total

        return {"BTC": float(w[0]), "ETH": float(w[1])}
