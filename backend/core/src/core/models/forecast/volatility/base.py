from abc import ABC, abstractmethod
import pandas as pd


class VolatilityModel(ABC):
    @abstractmethod
    def fit(self, returns: pd.Series) -> None:
        """Fit on pd.Series of log-returns indexed by UTC timestamp."""

    @abstractmethod
    def predict(self, returns: pd.Series) -> pd.Series:
        """
        Return out-of-sample conditional std-dev forecasts sigma_t > 0,
        aligned to returns.index.
        One-step-ahead: predict()[t] uses only information up to t-1.
        """
