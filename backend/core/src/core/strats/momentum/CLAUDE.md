# CLAUDE.md — `core/strats/momentum/` subagent briefing

## Role

You are implementing the **momentum strategy** for the BTC/ETH portfolio pipeline.
In trending / low-volatility regimes the orchestrator routes to this strategy.

Read `core/strats/CLAUDE.md` first for the shared architectural context.

---

## Module layout to implement

```
core/strats/momentum/
├── __init__.py
├── CLAUDE.md           ← this file
└── strategy.py         # MomentumStrategy
```

---

## strategy.py — `MomentumStrategy`

Implements `StrategyProtocol` from `core/strats/base.py`.

```python
class MomentumStrategy:
    def __init__(
        self,
        horizons: list[int] = [21, 63, 126, 252],  # TSMOM look-back windows (days)
        vol_window: int = 63,                       # EWMA / rolling vol look-back
        target_vol: float = 0.15,                   # annualised target portfolio vol (15 %)
        max_weight: float = 1.0,                    # abs weight cap per asset
    ): ...
```

### `fit(prices: pd.DataFrame) -> None`

```python
# Internally compute log-returns and store for signal generation.
# Fit a GJR-GARCH or EWMA vol model on each asset's return series.
# No parameters need to be estimated beyond vol — TSMOM is non-parametric.
```

Prefer `EWMAModel` from `core.models.forecast.volatility.ewma` for speed.
Fall back to a simple `rolling std` if the model is not available.

### `predict_signal(prices: pd.DataFrame) -> dict[str, float]`

1. Compute log-returns: `r = log(prices).diff()`.

2. For each asset `a` ∈ {BTC, ETH} and each horizon `L`:
   ```
   cum_ret_L = r[a].rolling(L).sum().shift(1)   # shift: no lookahead
   ```

3. Average signals across horizons:
   ```
   raw_signal[a] = mean over L of cum_ret_L[a]  (last value)
   ```

4. Vol-scale:
   ```
   sigma[a] = vol_model[a].predict(r[a]).iloc[-1]  # last bar's forecast
   # annualise: sigma_annual = sigma * sqrt(252)
   vol_scaled[a] = raw_signal[a] / (sigma_annual[a] + 1e-8)
   ```

5. Convert to weight:
   ```
   w[a] = clip(vol_scaled[a] / target_vol, -max_weight, max_weight)
   ```

6. Scale down if total notional exceeds 1:
   ```
   total = sum(|w[a]|)
   if total > 1: w = {a: w[a]/total for a in w}
   ```

7. Return `{"BTC": w["BTC"], "ETH": w["ETH"]}`.

All intermediate series use only data up to the last row of `prices`.

### Helper: `raw_signals(prices) -> pd.DataFrame`

Returns a DataFrame with columns `['BTC', 'ETH']`, each being the
multi-horizon averaged cumulative return (before vol-scaling), indexed
like `prices`. Useful for diagnostics.

---

## Testing

Place tests in `backend/core/tests/strats/momentum/`.

### `test_momentum_strategy.py`

```python
# 1. Monotone upward prices → both weights positive
# 2. Monotone downward prices → both weights negative
# 3. predict_signal always returns {"BTC": w1, "ETH": w2} with w in [-1, 1]
# 4. sum(|w|) <= 1 after normalisation
# 5. fit() then predict_signal() does not raise
# 6. MomentumStrategy satisfies StrategyProtocol (duck-type check)
```

---

## Imports allowed

```python
from core.strats.base import StrategyProtocol
from core.models.forecast.volatility.ewma import EWMAModel
import numpy as np
import pandas as pd
```

Optional (prefer EWMA for speed, but GARCH is acceptable):
```python
from core.models.forecast.volatility.garch import GJRGARCHModel
```

Do NOT import from `core/backtest/` or `core/evaluation/`.

---

## Key empirical constraints

- **Vol-scaling is mandatory** — procyclical deleveraging without it amplifies drawdowns.
  Always divide by `sigma_annual + 1e-8` to avoid division by zero.
- **Multi-horizon aggregation** reduces signal noise — default to all four horizons.
- **BTC typically has lower relative volatility** → receives larger vol-scaled weight;
  do not hard-code equal weights.
- **Weight cap at 1.0 per asset** prevents leverage blow-up in low-vol regimes.
