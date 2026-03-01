import numpy as np
import pandas as pd
from core.models.co_mov.base import CoMovModel


class BEKKModel(CoMovModel):
    """
    BEKK(1,1) multivariate GARCH model (Engle & Kroner 1995).

    Conditional covariance recursion:
        H_t = C'C + A' eps_{t-1} eps_{t-1}' A + B' H_{t-1} B

    Positive definiteness is guaranteed by the C'C construction.

    This implementation uses a simplified method-of-moments approach:
    - C from Cholesky of sample covariance (unconditional covariance anchor)
    - A = 0.1 * I  (small shock loading)
    - B = 0.85 * I (high persistence)
    """

    def __init__(self):
        self._C = None
        self._A = None
        self._B = None
        self._H0 = None

    def fit(self, returns: pd.DataFrame) -> None:
        """
        Fit BEKK(1,1) via simplified method-of-moments:
        - C from Cholesky of sample covariance
        - A = 0.1 * I (small shock loading)
        - B = 0.85 * I (high persistence)
        """
        cov = returns.cov().values
        self._C = np.linalg.cholesky(cov)
        n = returns.shape[1]
        self._A = 0.1 * np.eye(n)
        self._B = 0.85 * np.eye(n)
        self._H0 = cov.copy()
        self._returns = returns.copy()

    def predict(self, returns: pd.DataFrame) -> pd.DataFrame:
        """
        Run BEKK recursion. Returns 'correlation', 'lower_tail_dep', 'cov_BTC_ETH'.
        """
        eps = returns.values  # shape (T, 2)
        T = len(eps)
        H = self._H0.copy()
        C, A, B = self._C, self._A, self._B
        CC = C @ C.T

        correlations = []
        covs = []

        for t in range(T):
            # Record current H_t
            h11 = max(H[0, 0], 1e-8)
            h22 = max(H[1, 1], 1e-8)
            rho = H[0, 1] / np.sqrt(h11 * h22)
            rho = np.clip(rho, -0.9999, 0.9999)
            correlations.append(rho)
            covs.append(H[0, 1])

            # Update H_{t+1}
            e = eps[t:t + 1, :]  # (1, 2)
            outer = e.T @ e      # (2, 2)
            H = CC + A.T @ outer @ A + B.T @ H @ B
            # Ensure symmetry and positive definiteness
            H = 0.5 * (H + H.T)
            eigvals = np.linalg.eigvalsh(H)
            if eigvals.min() < 1e-8:
                H += (1e-8 - eigvals.min()) * np.eye(2)

        lower_tail = np.clip(correlations, 0, None)
        return pd.DataFrame({
            'correlation': correlations,
            'lower_tail_dep': lower_tail,
            'cov_BTC_ETH': covs,
        }, index=returns.index)
