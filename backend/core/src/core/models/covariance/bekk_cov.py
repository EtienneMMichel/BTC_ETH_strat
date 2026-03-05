from __future__ import annotations

import numpy as np
import pandas as pd

from core.models.co_mov.correlation.bekk import BEKKModel
from core.models.covariance.base import CovarianceModel


class BEKKCovModel(CovarianceModel):
    """Covariance model backed by a BEKK(1,1) recursion.

    Wraps :class:`~core.models.co_mov.correlation.bekk.BEKKModel` without
    modifying its public interface.  After fitting, the BEKK recursion is
    re-run independently to record H_t diagonal elements (variances).

    The output ``var_BTC`` / ``var_ETH`` are H_t[0,0] / H_t[1,1] and
    ``cov_BTC_ETH`` is H_t[0,1].  All columns are **not** shifted because
    the BEKK recursion records H_t *before* the update — i.e. the value at
    index ``t`` corresponds to the prediction made at ``t-1``.  This
    matches the no-lookahead convention of the other covariance models.
    """

    def __init__(self) -> None:
        self._bekk = BEKKModel()
        self._C: np.ndarray | None = None
        self._A: np.ndarray | None = None
        self._B: np.ndarray | None = None
        self._H0: np.ndarray | None = None

    def fit(self, returns: pd.DataFrame) -> None:
        """Fit the underlying BEKK model and copy its private parameters."""
        self._bekk.fit(returns)

        # Guard: verify private attributes were set by BEKKModel.fit()
        for attr in ("_C", "_A", "_B", "_H0"):
            if getattr(self._bekk, attr, None) is None:
                raise RuntimeError(
                    f"BEKKModel.fit() did not set '{attr}'; cannot build BEKKCovModel."
                )

        self._C = self._bekk._C.copy()
        self._A = self._bekk._A.copy()
        self._B = self._bekk._B.copy()
        self._H0 = self._bekk._H0.copy()

    def predict(self, returns: pd.DataFrame) -> pd.DataFrame:
        """Re-run the BEKK recursion and return H_t diagonal + off-diagonal.

        The value recorded at index ``t`` is H_t *before* updating with
        ``eps_t``, matching the no-lookahead convention: it reflects
        information available up to ``t-1``.
        """
        if self._C is None:
            raise RuntimeError("BEKKCovModel has not been fitted. Call fit() first.")

        eps = returns.values  # shape (T, 2)
        T = len(eps)
        H = self._H0.copy()
        C, A, B = self._C, self._A, self._B
        CC = C @ C.T

        var_btc_list: list[float] = []
        var_eth_list: list[float] = []
        cov_list: list[float] = []

        for t in range(T):
            # Record H_t (information from t-1)
            var_btc_list.append(max(H[0, 0], 1e-10))
            var_eth_list.append(max(H[1, 1], 1e-10))
            cov_list.append(float(H[0, 1]))

            # Update H_{t+1}
            e = eps[t : t + 1, :]  # (1, 2)
            outer = e.T @ e         # (2, 2)
            H = CC + A.T @ outer @ A + B.T @ H @ B
            H = 0.5 * (H + H.T)
            eigvals = np.linalg.eigvalsh(H)
            if eigvals.min() < 1e-8:
                H += (1e-8 - eigvals.min()) * np.eye(2)

        return pd.DataFrame(
            {
                "var_BTC": var_btc_list,
                "cov_BTC_ETH": cov_list,
                "var_ETH": var_eth_list,
            },
            index=returns.index,
        )
