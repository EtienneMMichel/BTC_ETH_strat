import pandas as pd

from core.models.forecast.volatility.base import VolatilityModel


class EWMAModel(VolatilityModel):
    """
    EWMA (RiskMetrics) volatility model.

    sigma^2_t = lam * sigma^2_{t-1} + (1 - lam) * epsilon^2_{t-1}

    Default lambda = 0.94 (RiskMetrics daily standard).
    """

    def __init__(self, lam: float = 0.94) -> None:
        self.lam = lam
        self._returns: pd.Series | None = None

    def fit(self, returns: pd.Series) -> None:
        """Store returns for use in predict()."""
        self._returns = returns

    def predict(self, returns: pd.Series) -> pd.Series:
        """
        Return one-step-ahead EWMA vol forecasts.

        Uses pandas ewm with com = lam / (1 - lam), then shifts by 1 so that
        predict()[t] reflects only information available up to t-1.
        """
        # com parameter: lam = com / (com + 1)  =>  com = lam / (1 - lam)
        com = self.lam / (1 - self.lam)
        variance = returns.ewm(com=com).var()

        # shift(1): forecast at t uses variance estimated from data up to t-1
        variance_shifted = variance.shift(1)

        # ewm().var() produces NaN for the first observation (needs >= 2 points),
        # so after shift(1) both index 0 and index 1 are NaN.
        # Fill all leading NaNs with the full-sample variance as a warm-start.
        unconditional_var = returns.var()
        variance_shifted = variance_shifted.fillna(unconditional_var)

        sigma = variance_shifted.clip(lower=0).pow(0.5)
        return sigma
