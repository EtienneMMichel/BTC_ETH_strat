from __future__ import annotations

import numpy as np
import pandas as pd

from core.models.expected_returns.base import ExpectedReturnsModel


class RollingMeanReturns(ExpectedReturnsModel):
    """Baseline expected-returns estimator: rolling mean of log-returns.

    Parameters
    ----------
    window:
        Rolling window in days (default 63 ≈ 3 months).
    min_periods:
        Minimum observations required to compute the mean (default 5).
        Earlier rows use an expanding window as fallback.
    """

    def __init__(self, window: int = 63, min_periods: int = 5) -> None:
        self.window = window
        self.min_periods = min_periods

    def fit(self, prices: pd.DataFrame) -> None:
        """No parameters to estimate."""

    def predict(self, prices: pd.DataFrame) -> pd.DataFrame:
        """Return one-step-ahead mean log-return estimates.

        Row ``t`` reflects only data available up to ``t-1`` (shift applied).
        """
        log_ret = np.log(prices).diff()

        roll = log_ret.rolling(self.window, min_periods=self.min_periods).mean()
        exp = log_ret.expanding(min_periods=self.min_periods).mean()
        mu = roll.fillna(exp).shift(1)

        return mu[["BTC", "ETH"]]
