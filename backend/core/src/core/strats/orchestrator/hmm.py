"""
HMM-based regime classifier for the BTC/ETH portfolio pipeline.

HMMRegimeClassifier uses a Gaussian HMM with n_components states to classify
market regimes. Unlike RegimeClassifier (threshold-based), this classifier uses
full-data smoothed posteriors (evaluation mode — lookahead allowed).

Features fed to the HMM:
  [log_ret_BTC, log_ret_ETH, realized_vol_BTC, realized_vol_ETH]
  (4-dim observations; realized vol = rolling std of log returns, window=5)
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd


class HMMRegimeClassifier:
    """
    Gaussian HMM regime classifier.

    Parameters
    ----------
    n_components : int
        Number of hidden states (regimes). Default 2.
    covariance_type : str
        HMM covariance structure. Default "full".
    n_iter : int
        Maximum EM iterations. Default 100.
    vol_window : int
        Rolling window for realized volatility features. Default 5.
    random_state : int
        Random seed for reproducibility. Default 42.
    """

    def __init__(
        self,
        n_components: int = 2,
        covariance_type: str = "full",
        n_iter: int = 100,
        vol_window: int = 5,
        random_state: int = 42,
    ) -> None:
        self.n_components = n_components
        self.covariance_type = covariance_type
        self.n_iter = n_iter
        self.vol_window = vol_window
        self.random_state = random_state
        self._model = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_features(self, prices: pd.DataFrame) -> tuple[np.ndarray, pd.Index]:
        """
        Build (n_obs, 4) feature matrix from prices.

        Returns
        -------
        X : np.ndarray, shape (n_obs, 4)
        valid_index : pd.Index aligned to X rows
        """
        log_ret = np.log(prices).diff()
        vol_btc = log_ret["BTC"].rolling(self.vol_window).std()
        vol_eth = log_ret["ETH"].rolling(self.vol_window).std()

        features = pd.DataFrame(
            {
                "log_ret_btc": log_ret["BTC"],
                "log_ret_eth": log_ret["ETH"],
                "vol_btc": vol_btc,
                "vol_eth": vol_eth,
            },
            index=prices.index,
        )
        # Drop rows with NaN (first row of log_ret + first vol_window rows)
        features = features.dropna()
        return features.values, features.index

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self, prices: pd.DataFrame) -> "HMMRegimeClassifier":
        """
        Fit the Gaussian HMM on historical prices.

        Parameters
        ----------
        prices : pd.DataFrame
            Columns: ["BTC", "ETH"] — close prices, chronologically ordered.

        Returns
        -------
        self
        """
        from hmmlearn.hmm import GaussianHMM  # lazy import

        X, _ = self._build_features(prices)
        model = GaussianHMM(
            n_components=self.n_components,
            covariance_type=self.covariance_type,
            n_iter=self.n_iter,
            random_state=self.random_state,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(X)
        self._model = model
        return self

    def predict(self, prices: pd.DataFrame) -> pd.Series:
        """
        Predict hard regime labels using Viterbi decoding.

        Parameters
        ----------
        prices : pd.DataFrame
            Columns: ["BTC", "ETH"].

        Returns
        -------
        pd.Series
            Index aligned to valid observations (NaN rows at start dropped).
            Values: "State 0", "State 1", …
        """
        if self._model is None:
            raise RuntimeError("Call fit() before predict().")
        X, valid_index = self._build_features(prices)
        state_seq = self._model.predict(X)
        labels = pd.Series(
            [f"State {s}" for s in state_seq],
            index=valid_index,
            name="regime",
        )
        return labels

    def predict_proba(self, prices: pd.DataFrame) -> pd.DataFrame:
        """
        Predict smoothed posterior probabilities using the forward-backward algorithm.

        Parameters
        ----------
        prices : pd.DataFrame
            Columns: ["BTC", "ETH"].

        Returns
        -------
        pd.DataFrame
            Shape (n_valid_obs, n_components).
            Columns: ["State 0", "State 1", …]
            Index aligned to valid observations.
        """
        if self._model is None:
            raise RuntimeError("Call fit() before predict_proba().")
        X, valid_index = self._build_features(prices)
        # posteriors: shape (n_obs, n_components)
        posteriors = self._model.predict_proba(X)
        columns = [f"State {i}" for i in range(self.n_components)]
        return pd.DataFrame(posteriors, index=valid_index, columns=columns)
