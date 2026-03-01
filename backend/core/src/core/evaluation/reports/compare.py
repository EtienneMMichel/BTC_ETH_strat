from __future__ import annotations
import numpy as np
import pandas as pd
from core.evaluation.metrics.vol import qlike, mse_vol, mae_vol


_LOSS_FN = {
    "qlike": qlike,
    "mse": mse_vol,
    "mae": mae_vol,
}


def comparison_table(
    forecasts: dict[str, pd.Series],
    realised: pd.Series,
    metrics: list[str] | None = None,
) -> pd.DataFrame:
    """
    Build a model comparison table.

    Parameters
    ----------
    forecasts : {model_name: forecast_series}
    realised  : realised values series
    metrics   : list of metric names from {"qlike", "mse", "mae"} (default all three)

    Returns
    -------
    pd.DataFrame with models as rows and metrics as columns.
    """
    if metrics is None:
        metrics = ["qlike", "mse", "mae"]

    rows = {}
    for name, fc in forecasts.items():
        row = {}
        for m in metrics:
            fn = _LOSS_FN.get(m)
            if fn is None:
                raise ValueError(f"Unknown metric '{m}'. Choose from {list(_LOSS_FN)}")
            row[m] = fn(fc, realised)
        rows[name] = row

    return pd.DataFrame(rows).T  # models as rows, metrics as columns


def diebold_mariano(
    forecast_a: pd.Series,
    forecast_b: pd.Series,
    realised: pd.Series,
    loss: str = "mse",
    h: int = 1,
) -> dict[str, float]:
    """
    Diebold-Mariano (1995) test of equal predictive accuracy.

    Positive DM stat → forecast A has higher loss (A is worse than B).
    Negative DM stat → forecast B has higher loss (B is worse than A).

    Uses Newey-West HAC variance for h > 1.

    Parameters
    ----------
    forecast_a, forecast_b : forecast series to compare
    realised               : realised values series
    loss                   : "mse" | "mae" | "qlike"
    h                      : forecast horizon (for HAC bandwidth = h - 1)

    Returns {'dm_stat': float, 'p_value': float}
    """
    fn = _LOSS_FN.get(loss)
    if fn is None:
        raise ValueError(f"Unknown loss '{loss}'. Choose from {list(_LOSS_FN)}")

    # Align all three series
    common = forecast_a.index.intersection(forecast_b.index).intersection(realised.index)
    fa = forecast_a.loc[common].values.astype(float)
    fb = forecast_b.loc[common].values.astype(float)
    rv = realised.loc[common].values.astype(float)

    # Per-observation loss differential d_t = L(a_t, rv_t) - L(b_t, rv_t)
    # We compute per-observation losses via the scalar formula
    if loss == "mse":
        da = (fa - rv) ** 2
        db = (fb - rv) ** 2
    elif loss == "mae":
        da = np.abs(fa - rv)
        db = np.abs(fb - rv)
    elif loss == "qlike":
        eps = 1e-10
        fa_c = np.clip(fa, eps, None)
        fb_c = np.clip(fb, eps, None)
        rv_c = np.clip(rv, eps, None)
        da = rv_c**2 / fa_c**2 - np.log(rv_c**2 / fa_c**2) - 1
        db = rv_c**2 / fb_c**2 - np.log(rv_c**2 / fb_c**2) - 1
    else:
        raise ValueError(f"Unknown loss '{loss}'")

    d = da - db  # loss differential
    T = len(d)
    if T == 0:
        return {"dm_stat": float("nan"), "p_value": float("nan")}

    d_bar = d.mean()

    # Newey-West variance with bandwidth = max(h-1, 0)
    bw = max(h - 1, 0)
    gamma0 = np.var(d, ddof=0)
    nw_var = gamma0
    for lag in range(1, bw + 1):
        weight = 1 - lag / (bw + 1)
        gamma_lag = np.mean((d[lag:] - d_bar) * (d[:-lag] - d_bar))
        nw_var += 2 * weight * gamma_lag

    nw_var = max(nw_var, 1e-16)
    dm_stat = d_bar / np.sqrt(nw_var / T)

    from scipy import stats as scipy_stats
    p_value = float(2 * scipy_stats.norm.sf(abs(dm_stat)))  # two-sided

    return {"dm_stat": float(dm_stat), "p_value": p_value}
