import numpy as np
import pandas as pd
from scipy.stats import linregress
from statsmodels.tsa.stattools import adfuller


class MeanReversionStrategy:
    def __init__(
        self,
        z_entry: float = 1.0,
        z_exit: float = 0.1,
        adf_pvalue: float = 0.05,
        max_drawdown: float = 0.10,
        var_window: int = 63,
        var_alpha: float = 0.05,
        max_weight: float = 0.5,
        max_half_life: float = 60.0,  # bars; above this, spread too slow to trade
    ):
        self.z_entry = z_entry
        self.z_exit = z_exit
        self.adf_pvalue_threshold = adf_pvalue
        self.max_drawdown = max_drawdown
        self.var_window = var_window
        self.var_alpha = var_alpha
        self.max_weight = max_weight
        self.max_half_life = max_half_life
        # Fitted attributes
        self.beta: float = 1.0
        self.intercept: float = 0.0
        self.mu_ou: float = 0.0
        self.sigma_ou: float = 1.0
        self.kappa: float = 0.1
        self.half_life: float = 7.0
        self.adf_pvalue_: float = 1.0
        self.spread_stationary: bool = False
        self._spread: pd.Series | None = None

    def fit(self, prices: pd.DataFrame) -> None:
        log_btc = np.log(prices["BTC"])
        log_eth = np.log(prices["ETH"])

        # Hedge ratio beta via OLS
        slope, intercept, *_ = linregress(log_btc.values, log_eth.values)
        self.beta = float(slope)
        self.intercept = float(intercept)

        # Spread
        X = log_eth - self.beta * log_btc
        self._spread = X.copy()

        # Require at least 20 bars for meaningful estimation
        if len(X) < 20:
            self.spread_stationary = False
            return

        # OU params via OLS on discretised form: dX = a + b * X_{t-1}
        dX = X.diff().dropna()
        X_lag = X.shift(1).dropna()
        # Align on common index
        common = dX.index.intersection(X_lag.index)
        dX = dX.loc[common]
        X_lag = X_lag.loc[common]

        slope_b, intercept_a, *_ = linregress(X_lag.values, dX.values)
        # b = -kappa * dt  (kappa > 0 means mean-reverting; b should be < 0)
        # a = kappa * mu * dt
        self.kappa = max(-slope_b, 1e-6)   # kappa = -b > 0
        self.mu_ou = float(intercept_a / (self.kappa + 1e-8))
        self.sigma_ou = max(float(dX.std()), 1e-8)
        self.half_life = np.log(2) / self.kappa

        # ADF test on spread
        adf_result = adfuller(X.dropna().values, maxlag=1, autolag=None)
        self.adf_pvalue_ = float(adf_result[1])
        self.spread_stationary = self.adf_pvalue_ < self.adf_pvalue_threshold

    def spread_zscore(self, prices: pd.DataFrame) -> float:
        """Return the current z-score of the spread. Used by the orchestrator."""
        log_btc = np.log(prices["BTC"].iloc[-1])
        log_eth = np.log(prices["ETH"].iloc[-1])
        X_last = log_eth - self.beta * log_btc
        return float((X_last - self.mu_ou) / (self.sigma_ou + 1e-8))

    def predict_signal(self, prices: pd.DataFrame) -> dict[str, float]:
        _zero: dict[str, float] = {"BTC": 0.0, "ETH": 0.0}

        # ADF gate
        if not self.spread_stationary:
            return _zero

        # Half-life gate
        if self.half_life > self.max_half_life:
            return _zero

        if self._spread is None or len(self._spread) < 20:
            return _zero

        # Drawdown gate (approximate from spread equity curve)
        spread_ret = self._spread.diff().fillna(0)
        # Scale spread moves to avoid huge swings in equity curve
        equity = (1 + spread_ret * 0.1).cumprod()
        rolling_max = equity.expanding().max()
        current_dd = float((equity.iloc[-1] / rolling_max.iloc[-1]) - 1)
        if current_dd < -self.max_drawdown:
            return _zero

        # VaR gate
        spread_changes = self._spread.diff().dropna()
        if len(spread_changes) >= self.var_window:
            recent = spread_changes.iloc[-self.var_window:]
            var_level = float(np.quantile(recent.values, self.var_alpha))
            current_change = float(spread_changes.iloc[-1])
            # var_level is negative (left tail); var_level * 1.5 is more negative
            # Gate fires if current change is extremely bad (worse than 1.5x VaR)
            if current_change < var_level * 1.5:
                return _zero

        # Z-score
        z = self.spread_zscore(prices)

        if z > self.z_entry:
            # Spread above mean -> short spread: short ETH, long BTC
            w_eth = -self.max_weight
            w_btc = float(np.clip(self.max_weight * self.beta, -self.max_weight, self.max_weight))
        elif z < -self.z_entry:
            # Spread below mean -> long spread: long ETH, short BTC
            w_eth = self.max_weight
            w_btc = float(np.clip(-self.max_weight * self.beta, -self.max_weight, self.max_weight))
        elif abs(z) < self.z_exit:
            return _zero
        else:
            # Between exit and entry thresholds -> hold flat (simplified)
            return _zero

        return {
            "BTC": float(np.clip(w_btc, -self.max_weight, self.max_weight)),
            "ETH": float(np.clip(w_eth, -self.max_weight, self.max_weight)),
        }
