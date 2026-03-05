from __future__ import annotations

from typing import Any, Type

import numpy as np
import pandas as pd

from core.models.forecast.price.base import PriceForecastModel
from core.models.forecast.price.momentum import TSMOMModel
from core.models.expected_returns.base import ExpectedReturnsModel


class SignalExpectedReturns(ExpectedReturnsModel):
    """Expected-returns estimator backed by a price-forecast model.

    Converts a directional signal into a daily expected return by scaling
    it by a rolling volatility estimate:

        mu_t(asset) = signal_t(asset) × daily_vol_t(asset) × scale

    All quantities are in daily units. The ``MarkowitzStrategy`` divides the
    annualised ``risk_free_rate`` by 252 before comparing with these estimates.

    Parameters
    ----------
    model_cls:
        Price-forecast model class to instantiate per asset
        (default :class:`~core.models.forecast.price.momentum.TSMOMModel`).
    vol_window:
        Rolling window for the daily-vol denominator (default 60).
    scale:
        Multiplicative scaling factor on the signal (default 1.0).
    model_kwargs:
        Keyword arguments forwarded to ``model_cls.__init__``.
    """

    def __init__(
        self,
        model_cls: Type[PriceForecastModel] | None = None,
        vol_window: int = 60,
        scale: float = 1.0,
        model_kwargs: dict[str, Any] | None = None,
    ) -> None:
        self._model_cls: Type[PriceForecastModel] = model_cls if model_cls is not None else TSMOMModel
        self.vol_window = vol_window
        self.scale = scale
        self._model_kwargs: dict[str, Any] = model_kwargs or {}
        self._models: dict[str, PriceForecastModel] = {}

    def fit(self, prices: pd.DataFrame) -> None:
        """Fit one price-forecast model per asset."""
        for asset in ["BTC", "ETH"]:
            m = self._model_cls(**self._model_kwargs)
            m.fit(prices[asset])
            self._models[asset] = m

    def predict(self, prices: pd.DataFrame) -> pd.DataFrame:
        """Return one-step-ahead daily expected log-returns.

        Steps per asset:
        1. signal_t  = model.predict(prices[asset])   (already shift(1))
        2. daily_vol = log(prices[asset]).diff().rolling(vol_window).std().shift(1)
        3. mu_t      = signal_t × daily_vol × scale
        """
        mu: dict[str, pd.Series] = {}

        for asset in ["BTC", "ETH"]:
            signal = self._models[asset].predict(prices[asset])

            log_ret = np.log(prices[asset]).diff()
            daily_vol = (
                log_ret.rolling(self.vol_window).std().shift(1)
            )
            # Expanding fallback for early observations
            daily_vol = daily_vol.fillna(log_ret.expanding().std())
            daily_vol = daily_vol.clip(lower=1e-8)

            mu[asset] = signal * daily_vol * self.scale

        return pd.DataFrame(mu, index=prices.index)[["BTC", "ETH"]]
