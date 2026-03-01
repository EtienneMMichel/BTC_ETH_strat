from __future__ import annotations

import numpy as np
import pandas as pd

from core.models.forecast.price.base import PriceForecastModel


class TSMOMModel(PriceForecastModel):
    """Single-horizon time-series momentum model (TSMOM).

    Parameters
    ----------
    lookback : int
        Number of past days used to compute the cumulative return signal.
        Default 252 (one trading year).
    vol_window : int
        Rolling window for the volatility denominator. Default 60.
    """

    def __init__(self, lookback: int = 252, vol_window: int = 60) -> None:
        self.lookback = lookback
        self.vol_window = vol_window
        self._log_returns: pd.Series | None = None

    # ------------------------------------------------------------------
    def fit(self, prices: pd.Series) -> None:
        """Compute and store log-returns. No parameters to estimate."""
        self._log_returns = np.log(prices).diff()

    # ------------------------------------------------------------------
    def _compute_signal(
        self,
        prices: pd.Series,
        vol_series: pd.Series | None = None,
    ) -> pd.Series:
        log_returns = np.log(prices).diff()

        # Rolling sum of past L returns, shifted so t uses [t-L, t-1].
        # Fall back to expanding sum for early periods where rolling is NaN.
        rolling_cum = log_returns.rolling(self.lookback).sum()
        expanding_cum = log_returns.expanding().sum()
        cum_ret = rolling_cum.fillna(expanding_cum).shift(1)

        if vol_series is not None:
            rolling_vol = vol_series.shift(1)
        else:
            rolling_vol = log_returns.rolling(self.vol_window).std().shift(1)
            # Fall back to expanding std where rolling is NaN
            rolling_vol = rolling_vol.fillna(log_returns.expanding().std())

        rolling_vol = rolling_vol.clip(lower=1e-8)
        signal = cum_ret / rolling_vol
        return signal

    # ------------------------------------------------------------------
    def predict(
        self,
        prices: pd.Series,
        vol_series: pd.Series | None = None,
    ) -> pd.Series:
        """Return the vol-scaled momentum signal aligned to prices.index."""
        return self._compute_signal(prices, vol_series)

    # ------------------------------------------------------------------
    def raw_signal(self, prices: pd.Series) -> pd.Series:
        """Return sign(cumulative return) — unscaled signal."""
        log_returns = np.log(prices).diff()
        cum_ret = log_returns.rolling(self.lookback).sum().shift(1)
        return cum_ret.apply(np.sign)


class MomentumModel(PriceForecastModel):
    """Multi-horizon TSMOM: average of individual TSMOMModel signals.

    Parameters
    ----------
    horizons : list[int]
        Look-back windows to average over. Default [21, 63, 126, 252].
    vol_window : int
        Rolling window for the volatility denominator. Default 60.
    """

    def __init__(
        self,
        horizons: list[int] | None = None,
        vol_window: int = 60,
    ) -> None:
        if horizons is None:
            horizons = [21, 63, 126, 252]
        self.horizons = horizons
        self.vol_window = vol_window
        self._models: list[TSMOMModel] = [
            TSMOMModel(lookback=h, vol_window=vol_window) for h in horizons
        ]

    # ------------------------------------------------------------------
    def fit(self, prices: pd.Series) -> None:
        """Fit each sub-model on the same price series."""
        for m in self._models:
            m.fit(prices)

    # ------------------------------------------------------------------
    def predict(
        self,
        prices: pd.Series,
        vol_series: pd.Series | None = None,
    ) -> pd.Series:
        """Return the average vol-scaled signal across all horizons.

        The very first observation(s) where no prior return exists are filled
        by back-propagating the first valid signal value (bfill), preserving
        the sign of the earliest available signal.
        """
        signals = pd.concat(
            [m.predict(prices, vol_series) for m in self._models], axis=1
        )
        # mean(skipna=True) averages available horizons at each date.
        avg = signals.mean(axis=1)
        # Back-fill the very first 1-2 NaN observations (no prior data exists)
        # with the first valid signal so the output has no NaN.
        avg = avg.bfill()
        avg.index = prices.index
        return avg

    # ------------------------------------------------------------------
    def raw_signal(self, prices: pd.Series) -> pd.Series:
        """Average raw (unscaled) signals across horizons."""
        signals = pd.concat(
            [m.raw_signal(prices) for m in self._models], axis=1
        )
        avg = signals.mean(axis=1)
        avg.index = prices.index
        return avg
