import numpy as np
import pandas as pd
from scipy.stats import norm, t as t_dist, kendalltau
from abc import ABC, abstractmethod


class Copula(ABC):
    @abstractmethod
    def fit(self, u: np.ndarray, v: np.ndarray) -> None:
        """Fit copula to PIT marginals u, v in (0, 1)."""

    @abstractmethod
    def lower_tail_dep(self) -> float:
        """Return lower tail dependence coefficient lambda_L."""

    @abstractmethod
    def upper_tail_dep(self) -> float:
        """Return upper tail dependence coefficient lambda_U."""


class GaussianCopula(Copula):
    """Gaussian copula: no tail dependence by construction."""

    def fit(self, u: np.ndarray, v: np.ndarray) -> None:
        # Convert uniform marginals to normal quantiles
        z1 = norm.ppf(np.clip(u, 1e-6, 1 - 1e-6))
        z2 = norm.ppf(np.clip(v, 1e-6, 1 - 1e-6))
        self.rho = np.corrcoef(z1, z2)[0, 1]
        self.rho = np.clip(self.rho, -0.9999, 0.9999)

    def lower_tail_dep(self) -> float:
        return 0.0

    def upper_tail_dep(self) -> float:
        return 0.0


class StudentTCopula(Copula):
    """
    Student-t copula: symmetric tail dependence.

    Lower/upper tail dependence:
        lambda = 2 * t_{nu+1}(-sqrt((nu+1)(1-rho)/(1+rho)))
    """

    def __init__(self, nu: float = 5.0):
        self.nu = nu
        self.rho = 0.0

    def fit(self, u: np.ndarray, v: np.ndarray) -> None:
        # Convert uniform marginals to t-quantiles
        z1 = t_dist.ppf(np.clip(u, 1e-6, 1 - 1e-6), df=self.nu)
        z2 = t_dist.ppf(np.clip(v, 1e-6, 1 - 1e-6), df=self.nu)
        self.rho = np.corrcoef(z1, z2)[0, 1]
        self.rho = np.clip(self.rho, -0.9999, 0.9999)

    def lower_tail_dep(self) -> float:
        nu = self.nu
        rho = self.rho
        if rho >= 1.0:
            return 1.0
        arg = -np.sqrt((nu + 1) * (1 - rho) / (1 + rho + 1e-8))
        return 2 * t_dist.cdf(arg, df=nu + 1)

    def upper_tail_dep(self) -> float:
        # Symmetric: upper tail dep equals lower tail dep
        return self.lower_tail_dep()


class ClaytonCopula(Copula):
    """
    Clayton copula: lower tail dependence, no upper tail dependence.

    Tail dependence:
        lambda_L = 2^(-1/theta)
        lambda_U = 0

    Parameter estimation via Kendall's tau:
        tau = theta / (theta + 2)  =>  theta = 2 * tau / (1 - tau)
    """

    def __init__(self):
        self.theta = 1.0

    def fit(self, u: np.ndarray, v: np.ndarray) -> None:
        tau, _ = kendalltau(u, v)
        tau = max(tau, 1e-6)  # ensure positive association
        self.theta = max(2 * tau / (1 - tau + 1e-8), 1e-4)

    def lower_tail_dep(self) -> float:
        # lambda_L = 2^(-1/theta)
        return 2 ** (-1.0 / self.theta)

    def upper_tail_dep(self) -> float:
        return 0.0


class RollingCopula:
    """
    Fits a copula on rolling windows and returns a time-series of tail dependence.

    Parameters
    ----------
    copula_class : type
        A Copula subclass (e.g. ClaytonCopula). A new instance is created per window.
    window : int
        Rolling window length in observations.
    """

    def __init__(self, copula_class, window: int = 126):
        self.copula_class = copula_class
        self.window = window

    def fit_predict(self, u: pd.Series, v: pd.Series) -> pd.DataFrame:
        """
        Fit the copula on each rolling window and collect tail dependence estimates.

        Returns
        -------
        pd.DataFrame with columns 'lower_tail_dep' and 'upper_tail_dep',
        indexed identically to u/v. Values are NaN for the burn-in period
        (first `window` observations).
        """
        n = len(u)
        lower = np.full(n, np.nan)
        upper = np.full(n, np.nan)

        u_arr = u.values
        v_arr = v.values

        for t in range(self.window, n):
            u_win = u_arr[t - self.window:t]
            v_win = v_arr[t - self.window:t]
            cop = self.copula_class()
            cop.fit(u_win, v_win)
            lower[t] = cop.lower_tail_dep()
            upper[t] = cop.upper_tail_dep()

        return pd.DataFrame({
            'lower_tail_dep': lower,
            'upper_tail_dep': upper,
        }, index=u.index)
