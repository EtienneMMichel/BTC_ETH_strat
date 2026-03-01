from abc import ABC, abstractmethod
import pandas as pd


class CoMovModel(ABC):
    @abstractmethod
    def fit(self, returns: pd.DataFrame) -> None:
        """
        Fit on DataFrame of log-returns with columns ['BTC', 'ETH'],
        indexed by UTC timestamp.
        """

    @abstractmethod
    def predict(self, returns: pd.DataFrame) -> pd.DataFrame:
        """
        Return out-of-sample estimates aligned to returns.index.
        Minimum required columns: ['correlation', 'lower_tail_dep'].
        """
