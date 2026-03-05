from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd


class CovarianceModel(ABC):
    """Abstract base class for 2×2 time-varying covariance models.

    Implementations estimate the conditional covariance matrix Σ_t for the
    BTC/ETH pair.  All models must respect the no-lookahead constraint:
    ``predict()[t]`` uses only data available up to ``t-1``.
    """

    @abstractmethod
    def fit(self, returns: pd.DataFrame) -> None:
        """Fit the model on a DataFrame of log-returns.

        Parameters
        ----------
        returns:
            DataFrame with columns ``['BTC', 'ETH']``, UTC-indexed.
        """

    @abstractmethod
    def predict(self, returns: pd.DataFrame) -> pd.DataFrame:
        """Return one-step-ahead covariance components.

        Output columns: ``['var_BTC', 'cov_BTC_ETH', 'var_ETH']``.
        All values are aligned to ``returns.index`` with ``shift(1)``
        applied so that row ``t`` reflects information from ``t-1``.
        ``var_BTC`` and ``var_ETH`` are always positive.
        """
