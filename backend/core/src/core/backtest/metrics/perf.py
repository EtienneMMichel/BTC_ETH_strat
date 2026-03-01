import numpy as np
import pandas as pd


def sharpe_ratio(
    returns: pd.Series,
    rf: float = 0.0,
    periods_per_year: int = 252,
) -> float:
    """Annualised Sharpe ratio."""
    excess = returns - rf / periods_per_year
    std = excess.std()
    if std == 0 or np.isnan(std):
        return 0.0
    return float(excess.mean() / std * np.sqrt(periods_per_year))


def max_drawdown(equity_curve: pd.Series) -> float:
    """
    Maximum drawdown from peak. Returns a non-positive float.
    equity_curve: portfolio value series starting at 1.0.
    """
    rolling_max = equity_curve.expanding().max()
    dd = (equity_curve / rolling_max) - 1.0
    return float(dd.min())


def calmar_ratio(
    returns: pd.Series,
    equity_curve: pd.Series,
    periods_per_year: int = 252,
) -> float:
    """Annualised return divided by absolute max drawdown."""
    ann_return = float(returns.mean() * periods_per_year)
    mdd = abs(max_drawdown(equity_curve))
    if mdd == 0:
        return np.nan
    return ann_return / mdd


def historical_var(
    returns: pd.Series,
    alpha: float = 0.05,
    window: int | None = None,
) -> float | pd.Series:
    """
    Historical VaR at level alpha (left tail, negative number).
    If window is given, returns a rolling pd.Series; otherwise a scalar.
    """
    if window is not None:
        return returns.rolling(window).quantile(alpha)
    return float(np.quantile(returns.dropna().values, alpha))


def expected_shortfall(
    returns: pd.Series,
    alpha: float = 0.05,
    window: int | None = None,
) -> float | pd.Series:
    """
    Historical Expected Shortfall (CVaR) at level alpha.
    Returns mean of returns below VaR. Negative number.
    """
    if window is not None:
        def _es(x):
            q = np.quantile(x, alpha)
            tail = x[x <= q]
            return tail.mean() if len(tail) > 0 else q
        return returns.rolling(window).apply(_es, raw=True)
    var = historical_var(returns, alpha)
    tail = returns[returns <= var]
    return float(tail.mean()) if len(tail) > 0 else var


def win_rate(trade_log: pd.DataFrame) -> float:
    """Fraction of closed trades with positive PnL. trade_log must have a 'pnl' column."""
    if len(trade_log) == 0:
        return float("nan")
    if "pnl" not in trade_log.columns:
        raise ValueError("trade_log must have a 'pnl' column")
    return float((trade_log["pnl"] > 0).mean())


def compute_all(
    equity_curve: pd.Series,
    trade_log: pd.DataFrame,
    periods_per_year: int = 252,
) -> dict[str, float]:
    """Convenience wrapper computing all metrics from an equity curve + trade log."""
    returns = equity_curve.pct_change().dropna()
    result = {
        "sharpe_ratio": sharpe_ratio(returns, periods_per_year=periods_per_year),
        "max_drawdown": max_drawdown(equity_curve),
        "calmar_ratio": calmar_ratio(returns, equity_curve, periods_per_year),
        "historical_var_5pct": historical_var(returns, alpha=0.05),
        "expected_shortfall_5pct": expected_shortfall(returns, alpha=0.05),
    }
    if len(trade_log) > 0 and "pnl" in trade_log.columns:
        result["win_rate"] = win_rate(trade_log)
    return result
