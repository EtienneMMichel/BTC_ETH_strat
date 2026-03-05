from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd


class ExpectedReturnsModel(ABC):
    """Abstract base class for daily expected-return estimators.

    Implementations wrap price-forecast models to produce per-asset daily
    expected log-return estimates used by the Markowitz optimiser.

    All models must respect the no-lookahead constraint:
    ``predict()[t]`` uses only data available up to ``t-1``.
    """

    @abstractmethod
    def fit(self, prices: pd.DataFrame) -> None:
        """Fit internal models on historical prices.

        Parameters
        ----------
        prices:
            DataFrame with columns ``['BTC', 'ETH']`` (close prices),
            UTC-indexed.
        """

    @abstractmethod
    def predict(self, prices: pd.DataFrame) -> pd.DataFrame:
        """Return one-step-ahead expected daily log-returns.

        Output columns: ``['BTC', 'ETH']``.
        Row ``t`` reflects only information available at ``t-1``.
        """
