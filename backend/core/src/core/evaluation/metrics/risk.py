from __future__ import annotations
import numpy as np
import pandas as pd
from scipy import stats


def kupiec_test(
    violations: pd.Series,
    alpha: float,
) -> dict[str, float]:
    """
    Kupiec (1995) unconditional coverage (POF) test.

    violations: boolean Series, True where return < -VaR (i.e. VaR was violated)
    alpha: nominal coverage level (e.g. 0.05 for 95% VaR)

    Returns {'lr_stat': float, 'p_value': float, 'violation_rate': float}
    """
    v = violations.astype(int).values
    T = len(v)
    n = v.sum()  # number of violations

    if T == 0:
        return {"lr_stat": float("nan"), "p_value": float("nan"), "violation_rate": float("nan")}

    p_hat = n / T
    violation_rate = float(p_hat)

    # LR_uc = -2 * [log L(alpha) - log L(p_hat)]
    # log L(pi) = n*log(pi) + (T-n)*log(1-pi)
    eps = 1e-10
    p_hat_c = float(np.clip(p_hat, eps, 1 - eps))
    alpha_c = float(np.clip(alpha, eps, 1 - eps))

    ll_null = n * np.log(alpha_c) + (T - n) * np.log(1 - alpha_c)
    ll_alt = n * np.log(p_hat_c) + (T - n) * np.log(1 - p_hat_c) if 0 < n < T else 0.0

    lr_stat = -2 * (ll_null - ll_alt)
    lr_stat = max(lr_stat, 0.0)  # numerical guard
    p_value = float(stats.chi2.sf(lr_stat, df=1))

    return {
        "lr_stat": float(lr_stat),
        "p_value": p_value,
        "violation_rate": violation_rate,
    }


def christoffersen_test(
    violations: pd.Series,
    alpha: float,
) -> dict[str, float]:
    """
    Christoffersen (1998) conditional coverage test.
    LR_cc = LR_uc + LR_ind ~ chi^2(2) under H0.

    Returns {'lr_stat': float, 'p_value': float, 'lr_uc': float, 'lr_ind': float}
    """
    v = violations.astype(int).values
    T = len(v)

    uc = kupiec_test(violations, alpha)
    lr_uc = uc["lr_stat"]

    # Transition counts for independence test
    # n_ij = count of (v_{t-1}=i, v_t=j)
    n00 = int(((v[:-1] == 0) & (v[1:] == 0)).sum())
    n01 = int(((v[:-1] == 0) & (v[1:] == 1)).sum())
    n10 = int(((v[:-1] == 1) & (v[1:] == 0)).sum())
    n11 = int(((v[:-1] == 1) & (v[1:] == 1)).sum())

    eps = 1e-10
    # Transition probabilities under alternative (unrestricted)
    pi01 = n01 / (n00 + n01 + eps)
    pi11 = n11 / (n10 + n11 + eps)
    pi = (n01 + n11) / (n00 + n01 + n10 + n11 + eps)

    pi01_c = np.clip(pi01, eps, 1 - eps)
    pi11_c = np.clip(pi11, eps, 1 - eps)
    pi_c = np.clip(pi, eps, 1 - eps)

    ll_ind = (
        n00 * np.log(1 - pi01_c) + n01 * np.log(pi01_c)
        + n10 * np.log(1 - pi11_c) + n11 * np.log(pi11_c)
    )
    ll_null_ind = (
        (n00 + n10) * np.log(1 - pi_c) + (n01 + n11) * np.log(pi_c)
    )

    lr_ind = max(-2 * (ll_null_ind - ll_ind), 0.0)
    lr_cc = lr_uc + lr_ind
    p_value = float(stats.chi2.sf(lr_cc, df=2))

    return {
        "lr_stat": float(lr_cc),
        "p_value": p_value,
        "lr_uc": float(lr_uc),
        "lr_ind": float(lr_ind),
    }
