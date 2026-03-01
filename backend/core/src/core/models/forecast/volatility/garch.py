import pandas as pd
from arch import arch_model

from core.models.forecast.volatility.base import VolatilityModel


class GARCHModel(VolatilityModel):
    """Standard GARCH(1,1) model."""

    def fit(self, returns: pd.Series) -> None:
        model = arch_model(returns, vol="Garch", p=1, q=1)
        self._res = model.fit(disp="off")
        self.aic = self._res.aic
        self.bic = self._res.bic

    def predict(self, returns: pd.Series) -> pd.Series:
        cond_vol = pd.Series(
            self._res.conditional_volatility,
            index=returns.index,
        )
        # shift(1): predict()[t] = fitted_sigma[t-1] — uses only info up to t-1
        result = cond_vol.shift(1)
        # Fill the first NaN with the unconditional (sample) std dev
        result.iloc[0] = returns.std()
        return result


class GJRGARCHModel(VolatilityModel):
    """GJR-GARCH(1,1,1) model with asymmetric leverage effect."""

    def fit(self, returns: pd.Series) -> None:
        model = arch_model(returns, vol="Garch", p=1, o=1, q=1)
        self._res = model.fit(disp="off")
        self.aic = self._res.aic
        self.bic = self._res.bic

    def predict(self, returns: pd.Series) -> pd.Series:
        cond_vol = pd.Series(
            self._res.conditional_volatility,
            index=returns.index,
        )
        result = cond_vol.shift(1)
        result.iloc[0] = returns.std()
        return result


class EGARCHModel(VolatilityModel):
    """EGARCH(1,1) model (Nelson 1991) — log-variance specification."""

    def fit(self, returns: pd.Series) -> None:
        model = arch_model(returns, vol="EGARCH", p=1, q=1)
        self._res = model.fit(disp="off")
        self.aic = self._res.aic
        self.bic = self._res.bic

    def predict(self, returns: pd.Series) -> pd.Series:
        cond_vol = pd.Series(
            self._res.conditional_volatility,
            index=returns.index,
        )
        result = cond_vol.shift(1)
        result.iloc[0] = returns.std()
        return result
