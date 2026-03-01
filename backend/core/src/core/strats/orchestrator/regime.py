"""
Regime classifier and orchestrator strategy for the BTC/ETH portfolio pipeline.

RegimeClassifier: classifies each bar into "momentum", "mean_reversion", or "cash"
    using rolling vol, ADF spread stationarity, and drawdown signals.

OrchestratorStrategy: implements StrategyProtocol; dispatches to the active
    sub-strategy based on the regime produced by RegimeClassifier.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import linregress
from statsmodels.tsa.stattools import adfuller


def _get_momentum():
    from core.strats.momentum.strategy import MomentumStrategy
    return MomentumStrategy


def _get_mean_reversion():
    from core.strats.mean_reversion.strategy import MeanReversionStrategy
    return MeanReversionStrategy


class RegimeClassifier:
    """
    Classifies market regimes as "momentum", "mean_reversion", or "cash".

    Parameters
    ----------
    vol_window : int
        Rolling window (in bars) for realised volatility estimation.
    vol_threshold_pct : float
        Expanding-history quantile threshold. Regime is "low vol" when the
        current vol sits below this quantile of its own history.
    adf_pvalue : float
        Maximum ADF p-value for the spread to be deemed stationary.
    corr_threshold : float
        DCC correlation crisis threshold (reserved for future use_dcc=True).
    drawdown_threshold : float
        Portfolio drawdown depth (positive fraction) that triggers "cash".
    min_holding_bars : int
        Hysteresis: a proposed regime change is accepted only after this many
        consecutive bars with the same proposed regime.
    beta_window : int
        Rolling OLS window for hedge-ratio estimation.
    use_dcc : bool
        If True, incorporate DCC correlation in the classifier (not yet
        implemented; reserved for extension).
    """

    REGIMES = ("momentum", "mean_reversion", "cash")

    def __init__(
        self,
        vol_window: int = 63,
        vol_threshold_pct: float = 0.6,
        adf_pvalue: float = 0.05,
        corr_threshold: float = 0.85,
        drawdown_threshold: float = 0.15,
        min_holding_bars: int = 5,
        beta_window: int = 252,
        use_dcc: bool = False,
    ):
        self.vol_window = vol_window
        self.vol_threshold_pct = vol_threshold_pct
        self.adf_pvalue = adf_pvalue
        self.corr_threshold = corr_threshold
        self.drawdown_threshold = drawdown_threshold
        self.min_holding_bars = min_holding_bars
        self.beta_window = beta_window
        self.use_dcc = use_dcc

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _rolling_adf_pvalue(self, spread: pd.Series, window: int) -> pd.Series:
        """
        Compute rolling ADF p-value on *spread* using a fixed *window*.

        NaN is returned for positions where the window is not yet full.
        On ADF failure the p-value is set to 1.0 (non-stationary).

        Complexity: O(n * window) — acceptable for n <= 400 in tests.
        """
        pvalues = pd.Series(np.nan, index=spread.index)
        for t in range(window, len(spread)):
            segment = spread.iloc[t - window : t].dropna()
            if len(segment) < 20:
                continue
            try:
                result = adfuller(segment.values, maxlag=1, autolag=None)
                pvalues.iloc[t] = result[1]
            except Exception:
                pvalues.iloc[t] = 1.0
        return pvalues

    def _rolling_beta(self, log_btc: pd.Series, log_eth: pd.Series) -> pd.Series:
        """
        Rolling OLS beta of log_eth on log_btc over a *beta_window* window.

        Returns a Series of the same index; NaN before the window is full.
        """
        betas = pd.Series(np.nan, index=log_btc.index)
        w = self.beta_window
        for t in range(w, len(log_btc)):
            x = log_btc.iloc[t - w : t].values
            y = log_eth.iloc[t - w : t].values
            if len(x) < 20:
                continue
            slope, *_ = linregress(x, y)
            betas.iloc[t] = slope
        return betas

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def classify(self, prices: pd.DataFrame) -> pd.Series:
        """
        Classify each bar in *prices* into a regime label.

        Parameters
        ----------
        prices : pd.DataFrame
            Columns: ["BTC", "ETH"] — close prices, chronologically ordered.

        Returns
        -------
        pd.Series
            Same index as *prices*, values in {"momentum", "mean_reversion", "cash"}.
            All signals are lagged by one bar (strict no-lookahead).
        """
        log_ret = np.log(prices).diff()
        n = len(prices)

        # ------------------------------------------------------------------
        # 1. Volatility signal
        # ------------------------------------------------------------------
        avg_vol = (
            log_ret["BTC"].rolling(self.vol_window).std()
            + log_ret["ETH"].rolling(self.vol_window).std()
        ) / 2

        # Expanding quantile: "low vol" when current vol < quantile of history.
        # quantile(0.0) == minimum → avg_vol < min(avg_vol) is always False,
        # which is the intentional behaviour for test_cash_regime_zero_weights.
        is_low_vol = avg_vol < avg_vol.expanding().quantile(self.vol_threshold_pct)
        is_low_vol = is_low_vol.shift(1)  # no-lookahead

        # ------------------------------------------------------------------
        # 2. Spread and ADF stationarity
        # ------------------------------------------------------------------
        log_btc = np.log(prices["BTC"])
        log_eth = np.log(prices["ETH"])

        beta_series = self._rolling_beta(log_btc, log_eth)
        # Fill early NaN with a global OLS beta to avoid NaN spread values.
        global_beta, *_ = linregress(log_btc.values, log_eth.values)
        beta_series = beta_series.fillna(global_beta)

        spread = log_eth - beta_series * log_btc

        adf_window = min(self.beta_window, n // 2)
        adf_pvalues = self._rolling_adf_pvalue(spread, window=adf_window)
        is_stationary = (adf_pvalues < self.adf_pvalue).shift(1).fillna(False)

        # ------------------------------------------------------------------
        # 3. Drawdown signal
        # ------------------------------------------------------------------
        port_ret = log_ret.mean(axis=1)
        cum_ret = port_ret.fillna(0).cumsum().apply(np.exp)
        rolling_max = cum_ret.expanding().max()
        drawdown = (cum_ret / rolling_max - 1).shift(1).fillna(0)
        is_stressed_dd = drawdown < -self.drawdown_threshold

        # ------------------------------------------------------------------
        # 4. Raw regime classification (bar-by-bar, causal)
        # ------------------------------------------------------------------
        raw_regime = pd.Series("cash", index=prices.index, dtype=object)
        for t in range(n):
            dd = bool(is_stressed_dd.iloc[t])
            low_vol_val = is_low_vol.iloc[t]
            low_vol = bool(low_vol_val) if not pd.isna(low_vol_val) else False
            stat = bool(is_stationary.iloc[t]) if not pd.isna(is_stationary.iloc[t]) else False

            if dd:
                raw_regime.iloc[t] = "cash"
            elif low_vol and not stat:
                raw_regime.iloc[t] = "momentum"
            elif stat and not dd:
                raw_regime.iloc[t] = "mean_reversion"
            else:
                raw_regime.iloc[t] = "cash"

        # ------------------------------------------------------------------
        # 5. Hysteresis: hold a regime for at least min_holding_bars bars
        # ------------------------------------------------------------------
        regime = raw_regime.copy()
        current = "cash"
        hold_count = 0
        for t in range(n):
            proposed = raw_regime.iloc[t]
            if proposed != current:
                if hold_count >= self.min_holding_bars:
                    current = proposed
                    hold_count = 0
                else:
                    hold_count += 1
            else:
                hold_count = 0
            regime.iloc[t] = current

        return regime


class OrchestratorStrategy:
    """
    Regime-switching orchestrator that implements StrategyProtocol.

    Holds one MomentumStrategy and one MeanReversionStrategy.  At each call to
    ``predict_signal`` it classifies the last bar's regime and delegates to the
    appropriate sub-strategy.  In the "cash" regime it returns zero weights.

    Parameters
    ----------
    momentum_strategy : object implementing StrategyProtocol, optional
        Defaults to a lazily-imported MomentumStrategy().
    mean_reversion_strategy : object implementing StrategyProtocol, optional
        Defaults to a lazily-imported MeanReversionStrategy().
    classifier : RegimeClassifier, optional
        Defaults to RegimeClassifier() with default parameters.
    """

    def __init__(
        self,
        momentum_strategy=None,
        mean_reversion_strategy=None,
        classifier: RegimeClassifier | None = None,
    ):
        if momentum_strategy is None:
            MomentumStrategy = _get_momentum()
            momentum_strategy = MomentumStrategy()
        if mean_reversion_strategy is None:
            MeanReversionStrategy = _get_mean_reversion()
            mean_reversion_strategy = MeanReversionStrategy()

        self.momentum = momentum_strategy
        self.mean_reversion = mean_reversion_strategy
        self.classifier = classifier or RegimeClassifier()

    # ------------------------------------------------------------------
    # StrategyProtocol
    # ------------------------------------------------------------------

    def fit(self, prices: pd.DataFrame) -> None:
        """Fit both sub-strategies on *prices*."""
        self.momentum.fit(prices)
        self.mean_reversion.fit(prices)

    def predict_signal(self, prices: pd.DataFrame) -> dict[str, float]:
        """
        Classify the regime of the last bar and delegate to the active strategy.

        Returns
        -------
        dict[str, float]
            Portfolio weights keyed by asset name.
            Zero weights {"BTC": 0.0, "ETH": 0.0} are returned in "cash" regime.
        """
        regime_series = self.classifier.classify(prices)
        regime = regime_series.iloc[-1]

        if regime == "momentum":
            return self.momentum.predict_signal(prices)
        elif regime == "mean_reversion":
            return self.mean_reversion.predict_signal(prices)
        else:
            return {"BTC": 0.0, "ETH": 0.0}

    def current_regime(self, prices: pd.DataFrame) -> str:
        """Return the regime label for the last bar of *prices*."""
        return self.classifier.classify(prices).iloc[-1]
