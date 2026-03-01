import numpy as np
import pandas as pd


def rogers_satchell(ohlcv: pd.DataFrame, annualize: bool = False) -> pd.Series:
    """
    Rogers-Satchell realized volatility estimator.

    Uses open/high/low/close log-prices — unbiased under drift.

    RS_t = (h-c)*(h-o) + (l-c)*(l-o)
    sigma_t = sqrt(max(RS_t, 0))

    Parameters
    ----------
    ohlcv : pd.DataFrame
        DataFrame with columns: open, high, low, close.
    annualize : bool
        If True, multiply by sqrt(252).

    Returns
    -------
    pd.Series of daily realized vol estimates.
    """
    h = np.log(ohlcv["high"])
    l = np.log(ohlcv["low"])
    c = np.log(ohlcv["close"])
    o = np.log(ohlcv["open"])

    rs = (h - c) * (h - o) + (l - c) * (l - o)
    # Clip to 0 to handle rare numerical noise producing tiny negatives
    sigma = np.sqrt(rs.clip(lower=0))

    if annualize:
        sigma = sigma * np.sqrt(252)

    return sigma


def yang_zhang(ohlcv: pd.DataFrame, annualize: bool = False, window: int = 21) -> pd.Series:
    """
    Yang-Zhang realized volatility estimator.

    Extends Rogers-Satchell to handle overnight jumps. Uses a rolling window
    to estimate variance components.

    sigma_YZ^2 = sigma_o^2 + k * sigma_c^2 + (1-k) * RS_rolling_mean
    where k = 0.34 / (1.34 + (window+1)/(window-1))

    Parameters
    ----------
    ohlcv : pd.DataFrame
        DataFrame with columns: open, high, low, close.
    annualize : bool
        If True, multiply by sqrt(252).
    window : int
        Rolling window size for variance estimation (default 21).

    Returns
    -------
    pd.Series of rolling Yang-Zhang realized vol estimates.
    """
    h = np.log(ohlcv["high"])
    l = np.log(ohlcv["low"])
    c = np.log(ohlcv["close"])
    o = np.log(ohlcv["open"])

    # Overnight return: log(open_t / close_{t-1})
    overnight_ret = o - c.shift(1)

    # Open-to-close return: log(close_t / open_t)
    open_close_ret = c - o

    # Rogers-Satchell per-day values
    rs = (h - c) * (h - o) + (l - c) * (l - o)
    rs = rs.clip(lower=0)

    # Rolling variance of overnight and open-to-close returns
    sigma_o2 = overnight_ret.rolling(window=window).var()
    sigma_c2 = open_close_ret.rolling(window=window).var()

    # Rolling mean of RS values
    rs_mean = rs.rolling(window=window).mean()

    # Optimal weight
    k = 0.34 / (1.34 + (window + 1) / (window - 1))

    sigma_yz2 = sigma_o2 + k * sigma_c2 + (1 - k) * rs_mean
    sigma_yz = np.sqrt(sigma_yz2.clip(lower=0))

    if annualize:
        sigma_yz = sigma_yz * np.sqrt(252)

    return sigma_yz
