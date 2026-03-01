from abc import ABC, abstractmethod
import pandas as pd


class PriceForecastModel(ABC):
    @abstractmethod
    def fit(self, prices: pd.Series) -> None:
        """Fit on pd.Series of prices indexed by UTC timestamp."""

    @abstractmethod
    def predict(self, prices: pd.Series) -> pd.Series:
        """
        Return out-of-sample signals aligned to prices.index.
        Signal: > 0 long bias, < 0 short bias, = 0 flat.
        One-step-ahead: predict()[t] uses only info up to t-1.
        """
