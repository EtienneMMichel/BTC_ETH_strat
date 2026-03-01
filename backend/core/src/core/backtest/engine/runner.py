from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np
import pandas as pd
from typing import Protocol


class StrategyProtocol(Protocol):
    def fit(self, prices: pd.DataFrame) -> None: ...
    def predict_signal(self, prices: pd.DataFrame) -> dict[str, float]: ...


@dataclass
class BacktestResult:
    equity_curve: pd.Series          # portfolio value over time, starting at 1.0
    trade_log: pd.DataFrame          # columns: timestamp, asset, qty, price, cost, pnl
    metrics: dict[str, float]        # from metrics/perf.py
    signals: pd.DataFrame            # target weights at each rebalance bar


class WalkForwardRunner:
    """
    Walk-forward / expanding-window backtest engine.

    At each rebalance step t (spaced rebalance_every bars apart, starting after min_train_bars):
    1. Fit strategy on prices[0:t]
    2. Predict signal (target weights) for bars [t, t+rebalance_every)
    3. Apply weights at bar t+1 (execution at next open is approximated by using close prices)
    4. Hold until next rebalance
    """

    def __init__(
        self,
        strategy: StrategyProtocol,
        min_train_bars: int = 252,
        rebalance_every: int = 21,
        fee_rate: float = 0.001,
        slippage: float = 0.0005,
        initial_capital: float = 1.0,
    ):
        self.strategy = strategy
        self.min_train_bars = min_train_bars
        self.rebalance_every = rebalance_every
        self.fee_rate = fee_rate
        self.slippage = slippage
        self.initial_capital = initial_capital

    def run(self, prices: pd.DataFrame) -> BacktestResult:
        """
        Run the backtest on a DataFrame of close prices with columns ['BTC', 'ETH'].
        Returns a BacktestResult.
        """
        from core.backtest.metrics.perf import compute_all

        n = len(prices)
        assets = list(prices.columns)

        # Portfolio state
        portfolio_value = self.initial_capital
        cash = portfolio_value
        positions: dict[str, float] = {a: 0.0 for a in assets}  # units held
        prev_weights: dict[str, float] = {a: 0.0 for a in assets}

        equity_values = []
        trade_records = []
        signal_records = []

        rebalance_steps = list(range(self.min_train_bars, n, self.rebalance_every))

        for t in range(n):
            price_row = prices.iloc[t]
            # Mark-to-market portfolio value
            port_val = cash + sum(positions[a] * price_row[a] for a in assets)
            portfolio_value = port_val
            equity_values.append(portfolio_value)

            if t in rebalance_steps:
                train_prices = prices.iloc[: t + 1]
                self.strategy.fit(train_prices)
                target_weights = self.strategy.predict_signal(train_prices)

                for a in assets:
                    w = float(target_weights.get(a, 0.0))
                    target_units = (portfolio_value * w) / price_row[a]
                    delta = target_units - positions[a]

                    if abs(delta) > 1e-10:
                        notional = abs(delta) * price_row[a]
                        cost = self.fee_rate * notional + self.slippage * notional
                        # Rough PnL: realised gain from closing delta position
                        pnl = delta * price_row[a] - cost
                        trade_records.append({
                            "timestamp": prices.index[t],
                            "asset": a,
                            "qty": delta,
                            "price": price_row[a],
                            "cost": cost,
                            "pnl": pnl,
                        })
                        positions[a] = target_units
                        cash -= delta * price_row[a] + cost * np.sign(delta)

                signal_records.append({
                    "timestamp": prices.index[t],
                    **{f"w_{a}": target_weights.get(a, 0.0) for a in assets},
                })
                prev_weights = target_weights

        equity_curve = pd.Series(
            equity_values,
            index=prices.index,
            name="equity",
        ) / self.initial_capital  # normalise to start at 1.0

        trade_log = pd.DataFrame(trade_records)
        if trade_log.empty:
            trade_log = pd.DataFrame(columns=["timestamp", "asset", "qty", "price", "cost", "pnl"])

        signals_df = pd.DataFrame(signal_records)
        if not signals_df.empty:
            signals_df = signals_df.set_index("timestamp")

        metrics = compute_all(equity_curve, trade_log)

        return BacktestResult(
            equity_curve=equity_curve,
            trade_log=trade_log,
            metrics=metrics,
            signals=signals_df,
        )
