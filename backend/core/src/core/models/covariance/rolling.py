from __future__ import annotations

import pandas as pd

from core.models.covariance.base import CovarianceModel


class RollingCovModel(CovarianceModel):
    """Baseline covariance model using a rolling sample covariance matrix.

    Parameters
    ----------
    window:
        Rolling window in days (default 63 ≈ 3 months).
    min_periods:
        Minimum observations required; earlier rows use an expanding
        window as fallback (default 5).
    """

    def __init__(self, window: int = 63, min_periods: int = 5) -> None:
        self.window = window
        self.min_periods = min_periods

    def fit(self, returns: pd.DataFrame) -> None:
        """No parameters to estimate; stores nothing."""

    def predict(self, returns: pd.DataFrame) -> pd.DataFrame:
        """Return rolling-window covariance components, shifted by 1 bar.

        For the early period where fewer than ``min_periods`` observations
        are available the expanding-window estimate is used as a fallback.
        """
        btc = returns["BTC"]
        eth = returns["ETH"]

        # Rolling estimates
        roll_var_btc = btc.rolling(self.window).var()
        roll_var_eth = eth.rolling(self.window).var()
        roll_cov = btc.rolling(self.window).cov(eth)

        # Expanding fallback (min_periods already applied implicitly)
        exp_var_btc = btc.expanding(min_periods=self.min_periods).var()
        exp_var_eth = eth.expanding(min_periods=self.min_periods).var()
        exp_cov = btc.expanding(min_periods=self.min_periods).cov(eth)

        var_btc = roll_var_btc.fillna(exp_var_btc)
        var_eth = roll_var_eth.fillna(exp_var_eth)
        cov_btc_eth = roll_cov.fillna(exp_cov)

        # Shift by 1: row t uses only data up to t-1
        var_btc = var_btc.shift(1)
        var_eth = var_eth.shift(1)
        cov_btc_eth = cov_btc_eth.shift(1)

        # Ensure positive variances (clip after shift; NaN stays NaN)
        var_btc = var_btc.clip(lower=1e-10)
        var_eth = var_eth.clip(lower=1e-10)

        return pd.DataFrame(
            {
                "var_BTC": var_btc,
                "cov_BTC_ETH": cov_btc_eth,
                "var_ETH": var_eth,
            },
            index=returns.index,
        )
