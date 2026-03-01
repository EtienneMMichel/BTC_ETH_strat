# CLAUDE.md — `core/strats/` subagent briefing

## Role

You are building the **strategy layer** of the BTC/ETH regime-switching portfolio.
This module sits above `core/models/` (which produces forecasts) and below `core/backtest/`
(which simulates P&L). It contains three subagents with distinct responsibilities:

| Subagent | Folder | Main idea |
|----------|--------|-----------|
| Regime orchestrator | `orchestrator/` | Classifies market state; routes capital to one strategy at a time |
| Momentum strategy | `momentum/` | Trend-following on BTC and ETH individually in low-vol regimes |
| Mean-reversion strategy | `mean_reversion/` | Spread trading on the BTC-ETH OU process in ranging regimes |

Each subfolder will receive its own deeper briefing (CLAUDE.md and eventually SKILL.md).
**This file is the shared prior** — read it first to understand how the three pieces fit before
diving into any one subfolder.

---

## The core idea in one paragraph

The market alternates between two broad states: **trending** (persistent directional moves,
low realised vol, correlation within normal range) and **ranging/stressed** (mean-reverting
spread, elevated vol, spiking correlation, drawdown pressure). The orchestrator detects which
state is active and routes to the matching strategy. The momentum strategy bets on continuation;
the mean-reversion strategy bets on the spread snapping back. Neither strategy knows about the
other — the orchestrator is the only entity that holds both simultaneously and decides which
one is live.

---

## Suggested folder layout

```
core/strats/
├── __init__.py
├── CLAUDE.md                  ← this file (shared prior)
├── base.py                    # StrategyProtocol — the contract core/backtest/ calls
├── orchestrator/
│   ├── __init__.py
│   ├── CLAUDE.md              # orchestrator subagent briefing (to be added)
│   └── regime.py              # regime classifier + dispatcher
├── momentum/
│   ├── __init__.py
│   ├── CLAUDE.md              # momentum subagent briefing (to be added)
│   └── strategy.py            # TSMOM strategy implementation
└── mean_reversion/
    ├── __init__.py
    ├── CLAUDE.md              # mean-reversion subagent briefing (to be added)
    └── strategy.py            # OU spread strategy implementation
```

---

## base.py — shared strategy protocol

All strategy classes must implement this interface so `core/backtest/engine/runner.py`
can call them uniformly without knowing which strategy is active:

```python
from typing import Protocol
import pandas as pd

class StrategyProtocol(Protocol):
    def fit(self, prices: pd.DataFrame) -> None:
        """
        Fit internal models on historical prices.
        prices: DataFrame with columns ['BTC', 'ETH'] (close prices), UTC-indexed.
        Called at each walk-forward step before predict_signal().
        """

    def predict_signal(self, prices: pd.DataFrame) -> dict[str, float]:
        """
        Return target portfolio weights for the next bar.
        Keys: asset names ('BTC', 'ETH'). Values: weights in [-1, 1].
        Weights sum to <= 1; remainder is cash. Negative = short.
        Uses only prices up to and including the last row — no lookahead.
        """
```

The orchestrator itself also implements `StrategyProtocol` — from the backtest engine's
perspective it is just another strategy. Internally it holds references to the sub-strategies
and switches between them.

---

## Prior for the orchestrator subagent

**Main idea**: classify the current regime using a small set of observable signals, then
return weights from the appropriate sub-strategy.

Key signals to consider (not prescriptive — the orchestrator CLAUDE.md will elaborate):
- **Volatility level**: rolling realised vol vs. long-run median → trending if low, stressed if high.
- **Drawdown**: current drawdown from peak → stressed if deep.
- **Spread stationarity**: ADF p-value on `X_t = log P_ETH − β log P_BTC` → mean-reverting if stationary.
- **Correlation spike**: DCC correlation vs. threshold → stressed if above (crisis co-movement).

Decision logic (threshold-based to start; Markov-switching as an extension):
```
if vol_regime == "low" and spread_stationary == False:
    active = momentum
elif spread_stationary == True and drawdown < threshold:
    active = mean_reversion
else:
    active = cash  # no position in ambiguous / extreme-stress regimes
```

Regime must be determined using only information available at bar `t` (no lookahead).
Hysteresis / minimum holding period prevents whipsawing between regimes.

---

## Prior for the momentum subagent

**Main idea**: in a trending market, the best predictor of tomorrow's direction is the sign
of recent cumulative returns. Scale position size by inverse volatility to keep risk constant.

Core signal (from `core/models/forecast/price/momentum.py`):
```
signal_t(asset) = (1 / σ_t) · Σ_{k=1}^{L} r_{t-k}
```
- Compute independently for BTC and ETH.
- Convert signal to a weight: `w_t = clip(signal_t / target_vol, -1, 1)`.
- Vol estimate `σ_t` comes from `core/models/forecast/volatility/`.
- BTC typically receives a larger weight due to lower relative risk (verified empirically).

The strategy is **inactive** (returns zero weights) when the orchestrator routes elsewhere.
It does not know about regimes — it always computes a signal; the orchestrator decides
whether to use it.

---

## Prior for the mean-reversion subagent

**Main idea**: the log-price spread `X_t = log P_ETH − β log P_BTC` behaves like an
Ornstein-Uhlenbeck process when co-integration holds. Trade the reversion to mean.

OU dynamics:
```
dX_t = κ (μ − X_t) dt + σ_OU dW_t
```
- `κ` (speed of reversion), `μ` (long-run mean), `σ_OU` (spread vol) are estimated by OLS
  on the discretised form: `ΔX_t = a + b X_{t-1} + ε_t` where `b = −κ Δt`, `a = κ μ Δt`.
- Half-life of reversion: `t_{1/2} = log(2) / κ`.
- Entry rule: go long spread when `X_t < μ − z · σ_OU`, short when `X_t > μ + z · σ_OU`.
  Default `z = 1.0`; exit at `|X_t − μ| < 0.1 · σ_OU`.
- Risk gates: suspend trading if current drawdown > threshold or rolling VaR is breached
  (drawdown and VaR computed via `core/backtest/metrics/perf.py`).
- `β` (hedge ratio) is estimated by OLS of `log P_ETH` on `log P_BTC`; re-estimated at
  each walk-forward step to handle structural breaks (ETH 2.0, ETF approvals).

Structural break awareness: if the ADF test on `X_t` fails (spread non-stationary),
the strategy returns zero weights and signals the orchestrator to switch to momentum.

---

## Dependency rules

```
core/strats/  →  core/models/          (consume forecasts, never modify)
core/strats/  →  (nothing else in core)
core/backtest/ →  core/strats/         (calls StrategyProtocol)
```

Strategies must not import from `core/backtest/` or `core/evaluation/`.
The only exception: strategies may use `core/backtest/metrics/perf.py` for internal
risk gates (drawdown, VaR) — these are read-only metric functions.

---

## Notes for future subagents

- Each subfolder (`orchestrator/`, `momentum/`, `mean_reversion/`) will get its own
  `CLAUDE.md` with detailed implementation specs and a `SKILL.md` with domain knowledge.
- When implementing a subfolder, read **this file first** to understand context, then the
  subfolder's own `CLAUDE.md` for precise specs.
- The `StrategyProtocol` in `base.py` is the only contract that must never change —
  it is the seam between `core/strats/` and `core/backtest/`.
