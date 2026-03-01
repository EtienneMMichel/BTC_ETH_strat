from __future__ import annotations
import numpy as np
import pandas as pd
from scipy import stats


def _align(forecast: pd.Series, realised: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    """Align two series on their common index, drop NaN."""
    common = forecast.index.intersection(realised.index)
    f = forecast.loc[common].values.astype(float)
    r = realised.loc[common].values.astype(float)
    mask = np.isfinite(f) & np.isfinite(r) & (f > 0) & (r > 0)
    return f[mask], r[mask]


def qlike(forecast: pd.Series, realised: pd.Series) -> float:
    """
    QLIKE loss: mean(sigma_t^2 / sigma_hat_t^2 - log(sigma_t^2 / sigma_hat_t^2) - 1)
    where sigma_t is realised vol and sigma_hat_t is forecast vol.
    Both inputs are std-dev series (not variance).
    Returns NaN if no valid pairs or any forecast <= 0.
    """
    f, r = _align(forecast, realised)
    if len(f) == 0:
        return float("nan")
    f2 = f ** 2
    r2 = r ** 2
    return float(np.mean(r2 / f2 - np.log(r2 / f2) - 1))


def mse_vol(forecast: pd.Series, realised: pd.Series) -> float:
    """Mean squared error between forecast and realised vol."""
    common = forecast.index.intersection(realised.index)
    f = forecast.loc[common].values.astype(float)
    r = realised.loc[common].values.astype(float)
    mask = np.isfinite(f) & np.isfinite(r)
    f, r = f[mask], r[mask]
    if len(f) == 0:
        return float("nan")
    return float(np.mean((f - r) ** 2))


def mae_vol(forecast: pd.Series, realised: pd.Series) -> float:
    """Mean absolute error between forecast and realised vol."""
    common = forecast.index.intersection(realised.index)
    f = forecast.loc[common].values.astype(float)
    r = realised.loc[common].values.astype(float)
    mask = np.isfinite(f) & np.isfinite(r)
    f, r = f[mask], r[mask]
    if len(f) == 0:
        return float("nan")
    return float(np.mean(np.abs(f - r)))


def mincer_zarnowitz(
    forecast: pd.Series, realised: pd.Series
) -> dict[str, float]:
    """
    Mincer-Zarnowitz regression: realised = alpha + beta * forecast + epsilon.
    Unbiased forecast: alpha ≈ 0, beta ≈ 1, high R².
    Returns {'alpha': ..., 'beta': ..., 'r_squared': ...}.
    """
    common = forecast.index.intersection(realised.index)
    f = forecast.loc[common].values.astype(float)
    r = realised.loc[common].values.astype(float)
    mask = np.isfinite(f) & np.isfinite(r)
    f, r = f[mask], r[mask]
    if len(f) < 2:
        return {"alpha": float("nan"), "beta": float("nan"), "r_squared": float("nan")}
    slope, intercept, r_value, _, _ = stats.linregress(f, r)
    return {
        "alpha": float(intercept),
        "beta": float(slope),
        "r_squared": float(r_value ** 2),
    }
