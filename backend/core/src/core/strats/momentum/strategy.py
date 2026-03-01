import numpy as np
import pandas as pd
from core.models.forecast.volatility.ewma import EWMAModel


class MomentumStrategy:
    """
    Time-series momentum (TSMOM) strategy for BTC/ETH.

    In trending / low-volatility regimes the orchestrator routes to this strategy.
    Signals are averaged over multiple look-back horizons and vol-scaled to keep
    risk constant across regimes.

    Implements StrategyProtocol from core.strats.base.
    """

    def __init__(
        self,
        horizons: list[int] | None = None,
        vol_window: int = 63,
        target_vol: float = 0.15,
        max_weight: float = 1.0,
    ):
        self.horizons = horizons or [21, 63, 126, 252]
        self.vol_window = vol_window
        self.target_vol = target_vol
        self.max_weight = max_weight
        self._vol_models: dict[str, EWMAModel] = {}

    def fit(self, prices: pd.DataFrame) -> None:
        """
        Fit internal EWMA vol models on each asset's log-return series.

        prices: DataFrame with columns ['BTC', 'ETH'] (close prices), UTC-indexed.
        Called at each walk-forward step before predict_signal().
        """
        log_ret = np.log(prices).diff()
        for asset in ["BTC", "ETH"]:
            m = EWMAModel()
            m.fit(log_ret[asset].dropna())
            self._vol_models[asset] = m
        self._prices = prices.copy()

    def raw_signals(self, prices: pd.DataFrame) -> pd.DataFrame:
        """
        Multi-horizon averaged cumulative returns (before vol scaling).

        Returns a DataFrame with columns ['BTC', 'ETH'] indexed like prices.
        Each column is the average of rolling(L).sum().shift(1) across all horizons.
        Useful for diagnostics.
        """
        log_ret = np.log(prices).diff()
        signals = {}
        for asset in ["BTC", "ETH"]:
            horizon_sigs = [
                log_ret[asset].rolling(L).sum().shift(1)
                for L in self.horizons
            ]
            signals[asset] = pd.concat(horizon_sigs, axis=1).mean(axis=1)
        return pd.DataFrame(signals)

    def predict_signal(self, prices: pd.DataFrame) -> dict[str, float]:
        """
        Return target portfolio weights for the next bar.

        Steps:
        1. Compute multi-horizon averaged cumulative log-returns (raw signal).
        2. Vol-scale each asset's raw signal by its annualised EWMA vol forecast.
        3. Clip to [-max_weight, max_weight].
        4. Normalise so sum(|w|) <= 1.

        Uses only prices up to and including the last row — no lookahead.
        """
        log_ret = np.log(prices).diff()
        raw = self.raw_signals(prices)

        weights = {}
        for asset in ["BTC", "ETH"]:
            raw_last = raw[asset].iloc[-1]
            if not np.isfinite(raw_last):
                raw_last = 0.0

            # Vol forecast (last bar, daily), annualise to match return units
            sigma_daily = self._vol_models[asset].predict(log_ret[asset].dropna()).iloc[-1]
            sigma_annual = sigma_daily * np.sqrt(252)
            sigma_annual = max(sigma_annual, 1e-8)

            vol_scaled = raw_last / sigma_annual
            weights[asset] = float(
                np.clip(vol_scaled / self.target_vol, -self.max_weight, self.max_weight)
            )

        # Normalise if total notional > 1
        total = sum(abs(w) for w in weights.values())
        if total > 1.0:
            weights = {a: w / total for a, w in weights.items()}

        return weights
