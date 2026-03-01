# CLAUDE.md — `core/backtest/` subagent briefing

## Role

You are building the **strategy-level backtesting harness** for the BTC/ETH regime-switching portfolio.
Your job is to consume strategy implementations from `core/strats/` and simulate their P&L over
historical BTC/ETH price data, producing equity curves, trade logs, and risk metrics.

**You do NOT implement regime-switching logic here.** Each strategy runs independently over a
full historical window; the regime orchestrator that selects among strategies lives in `core/strats/`.

---

## Module layout to implement

```
core/backtest/
├── __init__.py
├── CLAUDE.md                  ← this file
├── data/
│   ├── __init__.py
│   └── loader.py              # OHLCV loader for BTC and ETH
├── engine/
│   ├── __init__.py
│   └── runner.py              # walk-forward event loop
└── metrics/
    ├── __init__.py
    └── perf.py                # Sharpe, max drawdown, Calmar, VaR, trade stats
```

---

## data/loader.py

Responsibilities:
- Load daily or hourly BTC/ETH OHLCV data from a CSV or Parquet file.
- Return a `pd.DataFrame` indexed by UTC timestamp with columns:
  `open, high, low, close, volume` (lowercase). Multi-asset frames use a
  `pd.MultiIndex` on columns: `(asset, field)`.
- Validate: no duplicate timestamps, no all-NaN rows, monotonically increasing index.
- Expose a single public function:

```python
def load_ohlcv(path: str | Path, assets: list[str] = ["BTC", "ETH"]) -> pd.DataFrame:
    ...
```

No external data-fetching — this is a file-based loader only.

---

## engine/runner.py

Responsibilities:
- Implement a **walk-forward / expanding-window** backtest to avoid lookahead bias.
- The minimum training window is a constructor parameter (`min_train_bars`).
- At each rebalance step, the strategy is re-fit on `[0 : t]` and generates a signal for bar `t+1`.
- Positions are held until the next rebalance; rebalance frequency is configurable
  (`rebalance_every: int` bars).

**Transaction cost model** (flat fee + slippage):
```
cost_per_trade = fee_rate * notional + slippage * abs(delta_position) * price
```
Defaults: `fee_rate=0.001` (10 bps), `slippage=0.0005` (5 bps).

**Strategy interface contract** — the runner calls strategies via this protocol:

```python
class StrategyProtocol(Protocol):
    def fit(self, prices: pd.DataFrame) -> None: ...
    def predict_signal(self, prices: pd.DataFrame) -> dict[str, float]:
        """Return target weights, e.g. {"BTC": 0.6, "ETH": 0.4}."""
        ...
```

Weights must sum to ≤ 1 (remainder is cash); negative weights = short.

**Outputs** — `runner.run()` returns a `BacktestResult` dataclass:

```python
@dataclass
class BacktestResult:
    equity_curve: pd.Series          # portfolio value over time, starting at 1.0
    trade_log: pd.DataFrame          # columns: timestamp, asset, qty, price, cost
    metrics: dict[str, float]        # see metrics/perf.py
    signals: pd.DataFrame            # target weights at each rebalance bar
```

---

## metrics/perf.py

Implement the following functions; all accept a `pd.Series` of returns (daily or per-bar):

| Function | Formula / notes |
|---|---|
| `sharpe_ratio(returns, rf=0.0, periods_per_year=252)` | `mean / std * sqrt(periods)` — annualised |
| `max_drawdown(equity_curve)` | `min((V_t - max(V_0..t)) / max(V_0..t))` |
| `calmar_ratio(returns, equity_curve, periods_per_year=252)` | `annualised_return / abs(max_drawdown)` |
| `historical_var(returns, alpha=0.05)` | empirical quantile at `alpha` |
| `expected_shortfall(returns, alpha=0.05)` | mean of returns below VaR |
| `win_rate(trade_log)` | fraction of closed trades with positive PnL |

Expose a convenience wrapper:

```python
def compute_all(result: BacktestResult, periods_per_year: int = 252) -> dict[str, float]:
    ...
```

---

## Key design constraints (from the research study)

- **Never use forward-looking data.** The engine must enforce strict time-ordering; any signal
  computed at bar `t` may only use data up to and including bar `t`.
- **Volatility is clustered and regime-dependent.** The `historical_var` / `expected_shortfall`
  functions should support a `window` parameter for rolling estimation in addition to full-sample.
- **Transaction costs matter at high rebalance frequency.** Make `fee_rate` and `slippage`
  first-class parameters; the default run should show cost breakdown in `trade_log`.
- **BTC dominates optimal weights.** Tests should verify that an equal-weight baseline underperforms
  a vol-scaled strategy — this is documented in the study and should appear in example usage.

---

## Testing

Place tests in `backend/core/tests/backtest/`. Run with:
```bash
cd backend/core && poetry run pytest tests/backtest/
```

Minimum test coverage:
- `test_loader.py` — load a synthetic CSV, validate schema and types.
- `test_runner.py` — synthetic price series, single strategy, assert equity curve starts at 1.0,
  `max_drawdown <= 0`, Sharpe is finite, trade log is non-empty.
- `test_metrics.py` — unit-test each metric function against hand-calculated values.

---

## Interfaces you depend on (do not modify)

- `core/strats/` — strategy classes you will call via `StrategyProtocol` above.
  If a strategy class does not yet exist, mock it in tests.
- `core/models/` — you do NOT call model internals directly; strategies encapsulate model calls.
