import numpy as np
import pandas as pd
from scipy.optimize import minimize
from core.models.co_mov.base import CoMovModel


class ESCAViaRModel(CoMovModel):
    """
    ES-CAViaR model for joint tail risk estimation.

    Asymmetric-slope CAViaR specification:
        VaR_t(alpha) = beta0 + beta1*VaR_{t-1} + beta2*r_{t-1}^- + beta3*r_{t-1}^+
        ES_t(alpha)  = VaR_t(alpha) / alpha   (strict proportionality)

    where r^- = min(r, 0) and r^+ = max(r, 0).

    Parameters are estimated by minimising the pinball (quantile) loss at level alpha.

    Note: This model uses a single portfolio return formed as an equal-weight combination:
        r_port = 0.5 * r_BTC + 0.5 * r_ETH
    """

    def __init__(self, alpha: float = 0.05):
        self.alpha = alpha
        self.beta = None
        self._r_fit = None

    def _pinball_loss(self, beta: np.ndarray, r: np.ndarray) -> float:
        """Pinball (quantile) loss for CAViaR estimation."""
        VaR = np.zeros(len(r))
        VaR[0] = np.percentile(r, self.alpha * 100)
        for t in range(1, len(r)):
            r_neg = min(r[t - 1], 0)
            r_pos = max(r[t - 1], 0)
            VaR[t] = beta[0] + beta[1] * VaR[t - 1] + beta[2] * r_neg + beta[3] * r_pos

        residuals = r - VaR
        loss = np.where(
            residuals < 0,
            (self.alpha - 1) * residuals,
            self.alpha * residuals,
        )
        return loss.mean()

    def fit(self, returns: pd.DataFrame) -> None:
        """Fit CAViaR parameters on the equal-weight portfolio return."""
        r = (returns['BTC'] * 0.5 + returns['ETH'] * 0.5).values

        # Initial guess: small intercept, high persistence, asymmetric slope
        var_init = np.percentile(r, self.alpha * 100)
        beta0 = np.array([var_init * 0.05, 0.95, 0.1, 0.0])

        result = minimize(
            self._pinball_loss,
            beta0,
            args=(r,),
            method='Nelder-Mead',
            options={'maxiter': 2000, 'xatol': 1e-6, 'fatol': 1e-6},
        )
        self.beta = result.x
        self._r_fit = r

    def predict(self, returns: pd.DataFrame) -> pd.DataFrame:
        """
        Produce in-sample / out-of-sample VaR and ES estimates.

        Returns
        -------
        pd.DataFrame with columns:
            'var'            – conditional VaR (always negative, left tail)
            'es'             – conditional ES (always <= VaR)
            'correlation'    – zeros (not applicable; present for CoMovModel interface)
            'lower_tail_dep' – zeros (not applicable; present for CoMovModel interface)
        """
        r = (returns['BTC'] * 0.5 + returns['ETH'] * 0.5).values
        beta = self.beta

        VaR = np.zeros(len(r))
        VaR[0] = np.percentile(r, self.alpha * 100)
        for t in range(1, len(r)):
            r_neg = min(r[t - 1], 0)
            r_pos = max(r[t - 1], 0)
            VaR[t] = beta[0] + beta[1] * VaR[t - 1] + beta[2] * r_neg + beta[3] * r_pos

        # Ensure VaR is strictly negative (left-tail risk measure)
        VaR = np.minimum(VaR, -1e-8)
        # ES = VaR / alpha; since alpha < 1 and VaR < 0, ES is more negative than VaR
        ES = VaR / self.alpha

        return pd.DataFrame({
            'var': VaR,
            'es': ES,
            'correlation': np.zeros(len(r)),
            'lower_tail_dep': np.zeros(len(r)),
        }, index=returns.index)
