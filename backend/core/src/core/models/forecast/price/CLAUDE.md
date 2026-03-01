# CLAUDE.md — `core/models/forecast/price/` subagent briefing

## Role

You are the **price / direction forecasting specialist** for the BTC/ETH portfolio pipeline.
Your responsibility is to generate momentum signals and trend forecasts for BTC and ETH
individually. These signals are consumed by the momentum strategy in `core/strats/` and
scored by `core/evaluation/`.

A `SKILL.md` file will be added here later with deeper domain knowledge (signal construction,
look-back optimisation, statistical tests for predictability). Until then, use this file as
your primary brief.

---

## Module layout to implement

```
core/models/forecast/price/
├── __init__.py
├── CLAUDE.md                  ← this file
├── base.py                    # Abstract base class for all price/direction models
├── momentum.py                # Time-series momentum signals (TSMOM)
└── trend.py                   # Trend filters (EMA crossover, HP filter, Kalman smoother)
```

---

## base.py — shared interface

All price/direction models must implement this protocol so `core/evaluation/` can score them:

```python
from abc import ABC, abstractmethod
import pandas as pd

class PriceForecastModel(ABC):
    @abstractmethod
    def fit(self, prices: pd.Series) -> None:
        """
        Fit on a pd.Series of prices (not returns) indexed by UTC timestamp.
        Instantiate one model per asset (BTC or ETH).
        """

    @abstractmethod
    def predict(self, prices: pd.Series) -> pd.Series:
        """
        Return out-of-sample signals aligned to prices.index.
        Signal convention:
          > 0  → long bias (magnitude = confidence / strength)
          < 0  → short bias
          = 0  → no position / flat
        One-step-ahead: predict()[t] uses only information up to t-1.
        """
```

---

## momentum.py — Time-series momentum (TSMOM)

**Core signal** (Moskowitz, Ooi & Pedersen 2012, adapted for crypto):

```
signal_t = sign( Σ_{k=1}^{L} r_{t-k} )
```

where `r_{t-k}` are daily log-returns and `L` is the look-back window (default `L=252` days;
expose as constructor parameter).

**Volatility-scaled variant** (primary version to use):

```
signal_t = (1 / σ_t) · Σ_{k=1}^{L} r_{t-k}
```

where `σ_t` is a rolling or EWMA vol estimate. This normalises position size to target a
constant risk contribution.

**Multi-horizon variant**: compute signals at `L ∈ {21, 63, 126, 252}` days and average them.
Expose as `MomentumModel(horizons=[21, 63, 126, 252])`.

**Implementation notes**:
- `fit(prices)` computes and stores internal return series; no parameters to estimate.
- `predict(prices)` returns the vol-scaled signal series.
- Expose `raw_signal(prices)` (un-scaled sign) alongside `predict()` for diagnostics.

---

## trend.py — Trend filters

### EMA crossover

Classic fast/slow exponential moving average crossover:
```
signal_t = EMA(prices, span_fast) - EMA(prices, span_slow)
```
Defaults: `span_fast=50`, `span_slow=200`. Signal is positive when fast > slow (uptrend).

Normalise to `[-1, 1]` by dividing by a rolling std of the raw crossover series.

### Hodrick-Prescott (HP) filter

Decomposes prices into trend and cycle:
```
min_{τ} Σ (p_t - τ_t)² + λ Σ (Δ²τ_t)²
```
- `signal_t = sign(p_t - τ_t)`: positive when price is above trend (momentum).
- Default `λ = 1600` (standard quarterly frequency; for daily crypto use `λ = 6.25`).
- **One-sided (causal) version only** — compute HP filter recursively to avoid lookahead.

### Kalman smoother (trend component)

State-space model with random-walk trend:
```
level_{t} = level_{t-1} + slope_{t-1} + η_t     (η ~ N(0, Q))
slope_{t} = slope_{t-1} + ζ_t                    (ζ ~ N(0, R))
price_{t} = level_{t} + ε_t                       (ε ~ N(0, H))
```
- `fit(prices)` estimates `(Q, R, H)` via MLE.
- `predict(prices)` returns `slope_t` (positive = upward trend) as the signal.

---

## Key empirical constraints (from the research study)

- **Momentum strategy is activated in trending/low-vol regimes** — the signal is
  meaningless or harmful in high-vol/mean-reverting regimes; the regime orchestrator in
  `core/strats/` handles the switching; your job is to produce a clean signal.
- **Vol-scaling is mandatory** — procyclical deleveraging without vol-scaling amplifies
  drawdowns in crash regimes; always use the volatility-scaled TSMOM variant as default.
- **Multi-horizon aggregation reduces signal noise** — the multi-horizon `MomentumModel`
  should be the recommended production variant.
- **Structural breaks affect trend persistence** — the Kalman filter adapts dynamically
  and is preferred over fixed-parameter EMA crossover for non-stationary crypto prices.

---

## Testing

Place tests in `backend/core/tests/models/forecast/price/`. Minimum coverage:

- `test_momentum.py`:
  - Monotonically increasing price series → signal always positive.
  - Monotonically decreasing price series → signal always negative.
  - Random-walk prices → signal mean ≈ 0 over long sample.
  - Assert `predict().index == prices.index`.

- `test_trend.py`:
  - EMA crossover: `span_fast > span_slow` raises `ValueError`.
  - HP filter (causal): assert signal at `t` does not depend on `prices[t+1:]`
    (test by perturbing future prices and checking past signals are unchanged).
  - Kalman: assert `slope` series is finite and has same length as input.

- `test_base_contract.py` — any model implementing `PriceForecastModel` must pass:
  `predict(test).index == test.index`, signal is finite.

---

## Interfaces you expose (consumers)

- `core/strats/` — momentum strategy consumes `predict()` signal for position direction and sizing.
- `core/evaluation/` — scores `predict()` against realised next-bar returns
  (directional accuracy, Sharpe of signal × return).
- Do **not** import from `core/strats/`, `core/backtest/`, or `core/evaluation/`.
- Vol estimates used for scaling come from `core/models/forecast/volatility/` — import from
  there if needed, but keep the dependency optional (callers may inject a vol series externally).
