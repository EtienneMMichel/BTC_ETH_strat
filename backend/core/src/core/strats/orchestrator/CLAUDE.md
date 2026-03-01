# CLAUDE.md — `core/strats/orchestrator/` subagent briefing

## Role

You are the **regime-switching orchestrator** for the BTC/ETH portfolio pipeline.
Your job is to:
1. Classify the current market regime using observable signals (vol level, spread stationarity,
   drawdown depth, DCC correlation spike).
2. Dispatch to the correct sub-strategy (`MomentumStrategy` or `MeanReversionStrategy`).
3. Enforce hysteresis so the regime does not flip on every bar.

Read `core/strats/CLAUDE.md` first for the shared architectural context.

---

## Module layout to implement

```
core/strats/orchestrator/
├── __init__.py
├── CLAUDE.md           ← this file
└── regime.py           # RegimeClassifier + OrchestratorStrategy
```

---

## regime.py — two classes

### `RegimeClassifier`

Classifies each bar into one of three regimes: `"momentum"`, `"mean_reversion"`, `"cash"`.

**Constructor parameters:**
```python
RegimeClassifier(
    vol_window: int = 63,          # rolling window for realised vol
    vol_threshold_pct: float = 0.6,# regime is "low vol" if current vol < this quantile of history
    adf_pvalue: float = 0.05,      # spread is stationary if ADF p-value < this
    corr_threshold: float = 0.85,  # crisis if DCC correlation exceeds this
    drawdown_threshold: float = 0.15,  # cash if drawdown exceeds this
    min_holding_bars: int = 5,     # hysteresis: hold regime for at least N bars
)
```

**`classify(prices: pd.DataFrame) -> pd.Series`**

Returns a Series indexed like `prices` with values `"momentum"`, `"mean_reversion"`, or `"cash"`.
Uses only past information at each step (no lookahead).

Signal construction (strict causal, using `.shift(1)` where needed):

1. **Log-returns**: `log_ret = log(prices).diff()`

2. **Realised vol (rolling)**: `vol_BTC = log_ret['BTC'].rolling(vol_window).std()`
   — use the average of BTC and ETH rolling vol.

3. **Vol regime**: at each bar, compute `expanding` quantile of historical vol.
   `is_low_vol = (current_vol < vol.expanding().quantile(vol_threshold_pct)).shift(1)`

4. **Spread**: `X_t = log(prices['ETH']) - beta * log(prices['BTC'])`
   where `beta` is the OLS slope of `log(P_ETH)` on `log(P_BTC)` estimated on a 252-bar
   rolling window (use `scipy.stats.linregress`).
   Shift spread by 1 bar.

5. **ADF test on spread** (rolling, 252-bar window):
   Use `statsmodels.tsa.stattools.adfuller` on each rolling window.
   `is_stationary = (adf_p_value < adf_pvalue).shift(1)`

6. **Drawdown** (from peak of equity, approximated as cumulative returns of equal-weight portfolio):
   `cum_ret = (log_ret.mean(axis=1)).cumsum().apply(np.exp)`
   `rolling_max = cum_ret.expanding().max()`
   `drawdown = (cum_ret / rolling_max - 1).shift(1)`
   `is_stressed_dd = drawdown < -drawdown_threshold`

7. **Decision tree** (applied bar-by-bar after all signals are computed):
   ```
   if is_stressed_dd:
       regime = "cash"
   elif is_low_vol and not is_stationary:
       regime = "momentum"
   elif is_stationary and not is_stressed_dd:
       regime = "mean_reversion"
   else:
       regime = "cash"
   ```

8. **Hysteresis**: once a regime is active, keep it for at least `min_holding_bars` bars
   before allowing a switch.

**DCC correlation** is intentionally excluded from the classifier for now to keep
the dependency graph simple (DCC requires fitting which is expensive). It can be added
as an optional extension. The classifier should have a `use_dcc: bool = False` flag.

---

### `OrchestratorStrategy`

Implements `StrategyProtocol` from `core/strats/base.py`.

```python
class OrchestratorStrategy:
    def __init__(
        self,
        momentum_strategy,      # MomentumStrategy instance
        mean_reversion_strategy,# MeanReversionStrategy instance
        classifier: RegimeClassifier | None = None,
    ): ...

    def fit(self, prices: pd.DataFrame) -> None:
        """Fit all sub-strategies and the classifier."""

    def predict_signal(self, prices: pd.DataFrame) -> dict[str, float]:
        """
        Classify the last bar's regime, then delegate to the active sub-strategy.
        Returns zero weights {"BTC": 0.0, "ETH": 0.0} in "cash" regime.
        """
```

The orchestrator's `predict_signal` must:
1. Run `classifier.classify(prices)` to get the regime series.
2. Take the **last value** of the regime series as the current regime.
3. Return `momentum_strategy.predict_signal(prices)` or
   `mean_reversion_strategy.predict_signal(prices)` accordingly.
4. Return `{"BTC": 0.0, "ETH": 0.0}` in "cash" regime.

---

## Testing

Place tests in `backend/core/tests/strats/orchestrator/`.

### `test_regime.py`

```python
# Minimum tests:
# 1. Monotone trending prices → regime is "momentum" most of the time after warm-up
# 2. Mean-reverting spread → regime is "mean_reversion" most of the time
# 3. Hysteresis: regime should not change more than once every min_holding_bars bars
# 4. classify() output length matches prices length
# 5. All regime values are in {"momentum", "mean_reversion", "cash"}
```

### `test_orchestrator.py`

```python
# 1. predict_signal returns {"BTC": w1, "ETH": w2} always
# 2. Weights are in [-1, 1]
# 3. With trending prices, BTC weight > 0 (long bias)
# 4. OrchestratorStrategy implements StrategyProtocol
```

---

## Imports allowed

```python
from core.strats.base import StrategyProtocol
from core.strats.momentum.strategy import MomentumStrategy
from core.strats.mean_reversion.strategy import MeanReversionStrategy
import numpy as np
import pandas as pd
from scipy.stats import linregress
from statsmodels.tsa.stattools import adfuller
```

Do NOT import from `core/backtest/` or `core/evaluation/`.

---

## Key empirical constraints

- BTC-ETH correlation is **positive and time-varying** — the spread can be non-stationary
  for extended periods (ETH 2.0, BTC ETF events). The ADF gate is essential.
- Volatility clustering means regimes can persist for weeks — `min_holding_bars=5` is a
  minimum; real-world regimes last 20–60 bars. Hysteresis prevents noise-driven switching.
- The "cash" safety regime protects capital in ambiguous or extreme-stress states.
