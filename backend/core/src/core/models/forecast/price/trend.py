from __future__ import annotations

import numpy as np
import pandas as pd
from statsmodels.tsa.filters.hp_filter import hpfilter
from filterpy.kalman import KalmanFilter

from core.models.forecast.price.base import PriceForecastModel


class EMACrossover(PriceForecastModel):
    """EMA crossover trend model.

    Parameters
    ----------
    span_fast : int
        Span for the fast EMA. Must be strictly less than ``span_slow``.
    span_slow : int
        Span for the slow EMA.
    """

    def __init__(self, span_fast: int = 50, span_slow: int = 200) -> None:
        if span_fast >= span_slow:
            raise ValueError(
                f"span_fast ({span_fast}) must be < span_slow ({span_slow})"
            )
        self.span_fast = span_fast
        self.span_slow = span_slow

    # ------------------------------------------------------------------
    def fit(self, prices: pd.Series) -> None:
        """Nothing to estimate; validation already done in __init__."""
        pass

    # ------------------------------------------------------------------
    def predict(self, prices: pd.Series) -> pd.Series:
        """Return the normalised EMA crossover signal in [-1, 1], shifted by 1."""
        ema_fast = prices.ewm(span=self.span_fast, adjust=False).mean()
        ema_slow = prices.ewm(span=self.span_slow, adjust=False).mean()
        raw = ema_fast - ema_slow

        # Normalise by rolling std of the raw crossover series
        rolling_std = raw.rolling(self.span_slow).std().clip(lower=1e-8)
        signal = (raw / rolling_std).clip(-1.0, 1.0)

        # Shift by 1 for strict no-lookahead
        return signal.shift(1)


class HPFilter(PriceForecastModel):
    """One-sided (causal) Hodrick-Prescott filter.

    The causal version is implemented via an expanding window: for each time t,
    ``hpfilter`` is applied to ``prices[:t+1]`` and only the last value of the
    estimated trend is kept. This guarantees that ``signal[t]`` never depends on
    any price observed after time t.

    Parameters
    ----------
    lam : float
        Smoothing parameter lambda. Default 6.25 (daily frequency).
    """

    def __init__(self, lam: float = 6.25) -> None:
        self.lam = lam

    # ------------------------------------------------------------------
    def fit(self, prices: pd.Series) -> None:
        """Nothing to estimate."""
        pass

    # ------------------------------------------------------------------
    def predict(self, prices: pd.Series) -> pd.Series:
        """Return the causal HP-filter signal, shifted by 1 for no-lookahead."""
        trend = pd.Series(np.nan, index=prices.index, dtype=float)

        # hpfilter needs at least 3 data points
        for t in range(2, len(prices)):
            _, trend_arr = hpfilter(prices.iloc[: t + 1].values, lamb=self.lam)
            trend.iloc[t] = trend_arr[-1]

        # signal = sign(price - trend)
        signal = (prices - trend).apply(np.sign)

        # Shift by 1 so that signal[t] uses info up to t-1
        return signal.shift(1)


class KalmanTrend(PriceForecastModel):
    """Local linear trend model via Kalman filter.

    State vector: [level, slope].
    The slope series (forward-pass prior) is used as the directional signal.

    Parameters
    ----------
    q_level_scale : float
        Multiplier on the estimated level-noise variance.
    q_slope_scale : float
        Multiplier on the estimated slope-noise variance.
    r_scale : float
        Multiplier on the observation-noise variance.
    """

    def __init__(
        self,
        q_level_scale: float = 1.0,
        q_slope_scale: float = 1.0,
        r_scale: float = 0.1,
    ) -> None:
        self.q_level_scale = q_level_scale
        self.q_slope_scale = q_slope_scale
        self.r_scale = r_scale
        self._q_level: float | None = None
        self._q_slope: float | None = None
        self._r: float | None = None

    # ------------------------------------------------------------------
    def fit(self, prices: pd.Series) -> None:
        """Estimate noise variances from the data via simple heuristics."""
        diff1 = prices.diff().dropna()
        diff2 = prices.diff().diff().dropna()

        var_level = float(diff1.var()) if len(diff1) > 1 else 1.0
        var_slope = float(diff2.var()) if len(diff2) > 1 else 1.0

        self._q_level = self.q_level_scale * max(var_level, 1e-12)
        self._q_slope = self.q_slope_scale * max(var_slope, 1e-12)
        self._r = self.r_scale * max(var_level, 1e-12)

    # ------------------------------------------------------------------
    def predict(self, prices: pd.Series) -> pd.Series:
        """Run the Kalman forward pass; return the prior slope at each step."""
        if self._q_level is None:
            raise RuntimeError("Call fit() before predict().")

        kf = KalmanFilter(dim_x=2, dim_z=1)
        kf.F = np.array([[1.0, 1.0], [0.0, 1.0]])
        kf.H = np.array([[1.0, 0.0]])
        kf.Q = np.diag([self._q_level, self._q_slope])
        kf.R = np.array([[self._r]])
        kf.x = np.array([[float(prices.iloc[0])], [0.0]])
        kf.P = np.eye(2) * 1000.0

        slopes: list[float] = []
        for p in prices.values:
            kf.predict()
            # Prior slope before incorporating current observation = one-step-ahead
            slopes.append(float(kf.x[1, 0]))
            kf.update(np.array([[float(p)]]))

        return pd.Series(slopes, index=prices.index, dtype=float)
