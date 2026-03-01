from typing import Protocol
import pandas as pd


class StrategyProtocol(Protocol):
    def fit(self, prices: pd.DataFrame) -> None:
        """
        Fit internal models on historical prices.
        prices: DataFrame with columns ['BTC', 'ETH'] (close prices), UTC-indexed.
        Called at each walk-forward step before predict_signal().
        """

    def predict_signal(self, prices: pd.DataFrame) -> dict[str, float]:
        """
        Return target portfolio weights for the next bar.
        Keys: asset names ('BTC', 'ETH'). Values: weights in [-1, 1].
        Weights sum to <= 1; remainder is cash. Negative = short.
        Uses only prices up to and including the last row — no lookahead.
        """
