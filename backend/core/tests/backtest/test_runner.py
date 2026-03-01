import numpy as np
import pandas as pd
import pytest
from core.backtest.engine.runner import WalkForwardRunner, BacktestResult


class ConstantLongStrategy:
    """Always goes long 50% BTC, 50% ETH."""
    def fit(self, prices): pass
    def predict_signal(self, prices):
        return {"BTC": 0.5, "ETH": 0.5}


class ZeroStrategy:
    """Always returns zero weights (cash)."""
    def fit(self, prices): pass
    def predict_signal(self, prices):
        return {"BTC": 0.0, "ETH": 0.0}


def make_prices(n=400, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2022-01-01", periods=n, freq="D")
    btc = np.exp(np.cumsum(rng.standard_normal(n) * 0.02)) * 10000
    eth = np.exp(np.cumsum(rng.standard_normal(n) * 0.025)) * 500
    return pd.DataFrame({"BTC": btc, "ETH": eth}, index=idx)


def test_equity_starts_at_one():
    prices = make_prices()
    runner = WalkForwardRunner(ConstantLongStrategy(), min_train_bars=100, rebalance_every=21)
    result = runner.run(prices)
    assert result.equity_curve.iloc[0] == pytest.approx(1.0, abs=1e-9)


def test_equity_length_matches_prices():
    prices = make_prices()
    runner = WalkForwardRunner(ConstantLongStrategy(), min_train_bars=100, rebalance_every=21)
    result = runner.run(prices)
    assert len(result.equity_curve) == len(prices)


def test_result_is_backtest_result():
    prices = make_prices()
    runner = WalkForwardRunner(ConstantLongStrategy(), min_train_bars=100, rebalance_every=21)
    result = runner.run(prices)
    assert isinstance(result, BacktestResult)


def test_max_drawdown_non_positive():
    prices = make_prices()
    runner = WalkForwardRunner(ConstantLongStrategy(), min_train_bars=100, rebalance_every=21)
    result = runner.run(prices)
    assert result.metrics["max_drawdown"] <= 0


def test_sharpe_is_finite():
    prices = make_prices()
    runner = WalkForwardRunner(ConstantLongStrategy(), min_train_bars=100, rebalance_every=21)
    result = runner.run(prices)
    assert np.isfinite(result.metrics["sharpe_ratio"])


def test_trade_log_nonempty():
    prices = make_prices()
    runner = WalkForwardRunner(ConstantLongStrategy(), min_train_bars=100, rebalance_every=21)
    result = runner.run(prices)
    assert len(result.trade_log) > 0


def test_trade_log_columns():
    prices = make_prices()
    runner = WalkForwardRunner(ConstantLongStrategy(), min_train_bars=100, rebalance_every=21)
    result = runner.run(prices)
    for col in ["timestamp", "asset", "qty", "price", "cost", "pnl"]:
        assert col in result.trade_log.columns


def test_zero_strategy_no_trades():
    prices = make_prices()
    runner = WalkForwardRunner(ZeroStrategy(), min_train_bars=100, rebalance_every=21)
    result = runner.run(prices)
    # Zero strategy: no positions taken, equity stays at 1.0
    assert len(result.trade_log) == 0


def test_signals_dataframe():
    prices = make_prices()
    runner = WalkForwardRunner(ConstantLongStrategy(), min_train_bars=100, rebalance_every=21)
    result = runner.run(prices)
    assert isinstance(result.signals, pd.DataFrame)
    assert len(result.signals) > 0
