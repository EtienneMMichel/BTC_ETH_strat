from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit

from core.models.forecast.price.base import PriceForecastModel


class LogisticModel(PriceForecastModel):
    """
    Rolling-feature logistic regression → P(next bar up) ∈ [0,1].

    Features: rolling sum of log-returns at lookbacks [1, 5, 10, 21, 63] bars.
    Uses scipy.special.expit + scipy.optimize.minimize (L-BFGS-B).
    Output: sigmoid probability, shift(1) for no-lookahead.
    """

    def __init__(
        self,
        lookbacks: list[int] | None = None,
        C: float = 1.0,
    ) -> None:
        self.lookbacks: list[int] = lookbacks if lookbacks is not None else [1, 5, 10, 21, 63]
        self.C = C
        self._beta: np.ndarray | None = None

    def _build_features(self, log_returns: pd.Series) -> pd.DataFrame:
        feats = {f"mom_{lb}": log_returns.rolling(lb).sum() for lb in self.lookbacks}
        return pd.DataFrame(feats, index=log_returns.index)

    def fit(self, prices: pd.Series) -> None:
        log_ret = np.log(prices).diff()
        X_df = self._build_features(log_ret)
        # Binary target: 1 if next bar return > 0
        y = (log_ret.shift(-1) > 0).astype(float)
        mask = X_df.notna().all(axis=1) & y.notna()
        X = X_df.loc[mask].values
        y_arr = y.loc[mask].values

        k = X.shape[1]
        beta0 = np.zeros(k + 1)  # intercept + k weights

        def nll(beta: np.ndarray) -> float:
            Xb = beta[0] + X @ beta[1:]
            p = np.clip(expit(Xb), 1e-9, 1 - 1e-9)
            l2 = 0.5 / self.C * float(np.sum(beta[1:] ** 2))
            return -float(np.mean(y_arr * np.log(p) + (1 - y_arr) * np.log(1 - p))) + l2

        result = minimize(nll, beta0, method="L-BFGS-B")
        self._beta = result.x

    def predict(self, prices: pd.Series) -> pd.Series:
        if self._beta is None:
            raise RuntimeError("Call fit() before predict().")
        log_ret = np.log(prices).diff()
        X_df = self._build_features(log_ret)
        Xb = self._beta[0] + X_df.values @ self._beta[1:]
        prob = pd.Series(expit(Xb), index=prices.index, name="logistic")
        # shift(1): signal at t uses only data up to t-1
        return prob.shift(1)
