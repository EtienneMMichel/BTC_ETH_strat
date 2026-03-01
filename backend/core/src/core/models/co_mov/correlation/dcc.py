import numpy as np
import pandas as pd
from arch import arch_model
from core.models.co_mov.base import CoMovModel


class DCCModel(CoMovModel):
    """
    DCC-GARCH model (Engle 2002).

    Step 1: Fit univariate GARCH(1,1) to each return series.
    Step 2: Extract standardised residuals.
    Step 3: Run DCC dynamics:
        Q_t = (1 - a - b) * Q_bar + a * eps_{t-1} eps_{t-1}' + b * Q_{t-1}
        R_t = diag(Q_t)^{-1/2} Q_t diag(Q_t)^{-1/2}
    """

    def __init__(self, a: float = 0.05, b: float = 0.93):
        self.a = a
        self.b = b
        self._std_resid = None
        self._Q_bar = None

    def fit(self, returns: pd.DataFrame) -> None:
        """
        Step 1: Fit univariate GARCH(1,1) to each column.
        Step 2: Compute standardised residuals.
        Step 3: Estimate Q_bar as sample covariance of standardised residuals.
        """
        self._returns = returns.copy()
        std_resids = {}
        for col in returns.columns:
            res = arch_model(
                returns[col] * 100, vol='Garch', p=1, q=1, rescale=False
            ).fit(disp='off')
            cond_vol = res.conditional_volatility / 100
            cond_vol = cond_vol.clip(lower=1e-8)
            std_resids[col] = returns[col] / cond_vol

        self._std_resid = pd.DataFrame(std_resids)
        self._Q_bar = self._std_resid.cov().values  # 2x2 unconditional covariance

    def predict(self, returns: pd.DataFrame) -> pd.DataFrame:
        """
        Run DCC dynamics on standardised residuals.
        Returns DataFrame with 'correlation' and 'lower_tail_dep' columns.
        """
        eps = self._std_resid.values  # shape (T, 2)
        T = len(eps)
        Q = self._Q_bar.copy()
        correlations = []

        for t in range(T):
            # Compute R_t from Q_t
            q11 = max(Q[0, 0], 1e-8)
            q22 = max(Q[1, 1], 1e-8)
            rho = Q[0, 1] / np.sqrt(q11 * q22)
            rho = np.clip(rho, -0.9999, 0.9999)
            correlations.append(rho)

            # Update Q_{t+1}
            e = eps[t:t + 1, :]  # shape (1, 2)
            Q = (1 - self.a - self.b) * self._Q_bar + self.a * (e.T @ e) + self.b * Q

        corr_series = pd.Series(correlations, index=returns.index)
        # Lower tail dep: approximate using correlation (simplified).
        # For Gaussian DCC, lower_tail_dep = 0, but approximate with max(0, rho).
        lower_tail = corr_series.clip(lower=0)

        return pd.DataFrame({
            'correlation': corr_series,
            'lower_tail_dep': lower_tail,
        })
