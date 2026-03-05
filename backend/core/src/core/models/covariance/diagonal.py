from __future__ import annotations

import pandas as pd

from core.models.co_mov.correlation.dcc import DCCModel
from core.models.covariance.base import CovarianceModel
from core.models.forecast.volatility.ewma import EWMAModel
from core.models.forecast.volatility.base import VolatilityModel
from core.models.co_mov.base import CoMovModel


class DiagonalCovModel(CovarianceModel):
    """Two-step DCC covariance model (Engle 2002).

    Separates marginal volatility estimation from correlation estimation,
    allowing each component to be swapped independently.

    Parameters
    ----------
    vol_model_btc:
        Univariate volatility model for BTC (default ``EWMAModel()``).
    vol_model_eth:
        Univariate volatility model for ETH (default ``EWMAModel()``).
    corr_model:
        Co-movement model producing a ``'correlation'`` column
        (default ``DCCModel()``).
    """

    def __init__(
        self,
        vol_model_btc: VolatilityModel | None = None,
        vol_model_eth: VolatilityModel | None = None,
        corr_model: CoMovModel | None = None,
    ) -> None:
        self._vol_btc: VolatilityModel = vol_model_btc if vol_model_btc is not None else EWMAModel()
        self._vol_eth: VolatilityModel = vol_model_eth if vol_model_eth is not None else EWMAModel()
        self._corr: CoMovModel = corr_model if corr_model is not None else DCCModel()

    def fit(self, returns: pd.DataFrame) -> None:
        """Fit univariate vol models and the correlation model."""
        self._vol_btc.fit(returns["BTC"])
        self._vol_eth.fit(returns["ETH"])
        self._corr.fit(returns)

    def predict(self, returns: pd.DataFrame) -> pd.DataFrame:
        """Return DCC-based covariance components, shifted by 1 bar.

        Steps
        -----
        1. Predict σ_BTC and σ_ETH from the univariate vol models.
        2. Predict ρ from the correlation model.
        3. Assemble Σ_t = diag(σ) @ R_t @ diag(σ).
        """
        sigma_btc = self._vol_btc.predict(returns["BTC"])  # already shift(1)
        sigma_eth = self._vol_eth.predict(returns["ETH"])  # already shift(1)
        corr_df = self._corr.predict(returns)
        rho = corr_df["correlation"]

        # Variance is σ²; shift already applied by vol models
        var_btc = sigma_btc ** 2
        var_eth = sigma_eth ** 2
        cov_btc_eth = rho * sigma_btc * sigma_eth

        return pd.DataFrame(
            {
                "var_BTC": var_btc,
                "cov_BTC_ETH": cov_btc_eth,
                "var_ETH": var_eth,
            },
            index=returns.index,
        )
