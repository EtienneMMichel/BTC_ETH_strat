# CLAUDE.md — `core/strats/mean_reversion/` subagent briefing

## Role

You are implementing the **mean-reversion (spread) strategy** for the BTC/ETH portfolio.
The strategy models the log-price spread `X_t = log P_ETH − β log P_BTC` as an
Ornstein-Uhlenbeck process and trades its reversion to the long-run mean.

Read `core/strats/CLAUDE.md` first for the shared architectural context.

---

## Module layout to implement

```
core/strats/mean_reversion/
├── __init__.py
├── CLAUDE.md           ← this file
└── strategy.py         # MeanReversionStrategy
```

---

## strategy.py — `MeanReversionStrategy`

Implements `StrategyProtocol` from `core/strats/base.py`.

```python
class MeanReversionStrategy:
    def __init__(
        self,
        z_entry: float = 1.0,          # enter when |z-score| > z_entry
        z_exit: float = 0.1,           # exit when |z-score| < z_exit
        adf_pvalue: float = 0.05,      # ADF gate: only trade if p-value < this
        max_drawdown: float = 0.10,    # suspend trading if rolling drawdown > 10%
        var_window: int = 63,          # rolling window for VaR computation
        var_alpha: float = 0.05,       # VaR confidence level
        max_weight: float = 0.5,       # abs weight cap per asset (spread is leveraged)
    ): ...
```

### `fit(prices: pd.DataFrame) -> None`

1. **Estimate hedge ratio β**:
   OLS of `log(prices['ETH'])` on `log(prices['BTC'])`:
   ```python
   from scipy.stats import linregress
   slope, intercept, _, _, _ = linregress(
       np.log(prices['BTC'].values),
       np.log(prices['ETH'].values)
   )
   self.beta = slope
   self.intercept = intercept
   ```

2. **Compute spread**:
   ```python
   X = np.log(prices['ETH']) - self.beta * np.log(prices['BTC'])
   ```

3. **Estimate OU parameters** via OLS on discretised form:
   `ΔX_t = a + b * X_{t-1} + ε_t`
   ```python
   dX = X.diff().dropna()
   X_lag = X.shift(1).dropna()
   slope_b, intercept_a, _, _, _ = linregress(X_lag.values, dX.values)
   # b = -kappa * dt  (kappa > 0 means mean-reverting; b < 0)
   # a = kappa * mu * dt
   self.kappa = -slope_b            # speed of reversion (> 0)
   self.mu_ou = intercept_a / (-slope_b + 1e-8)  # long-run mean
   self.sigma_ou = dX.std()         # spread vol
   self.half_life = np.log(2) / (self.kappa + 1e-8)
   ```

4. **ADF test**:
   ```python
   from statsmodels.tsa.stattools import adfuller
   adf_result = adfuller(X.dropna(), maxlag=1, autolag=None)
   self.adf_pvalue = adf_result[1]
   self.spread_stationary = self.adf_pvalue < self.adf_pvalue_threshold
   ```
   Store the full spread series `self._spread = X`.

5. **Store** `prices`, `self._log_ret` for risk gate computation.

### `predict_signal(prices: pd.DataFrame) -> dict[str, float]`

1. **ADF gate**: if `not self.spread_stationary`:
   return `{"BTC": 0.0, "ETH": 0.0}`.

2. **Risk gate — drawdown**:
   Compute the rolling P&L of the spread position (from `self._portfolio_returns` if
   stored during previous calls, or approximate via simple spread returns).
   Approximate:
   ```python
   spread_ret = self._spread.diff()
   equity = (1 + spread_ret.fillna(0)).cumprod()
   rolling_max = equity.expanding().max()
   current_drawdown = (equity.iloc[-1] / rolling_max.iloc[-1]) - 1
   if current_drawdown < -self.max_drawdown:
       return {"BTC": 0.0, "ETH": 0.0}
   ```

3. **Risk gate — VaR**:
   ```python
   spread_ret_window = self._spread.diff().dropna().iloc[-self.var_window:]
   var = np.quantile(spread_ret_window, self.var_alpha)  # negative number
   # If current spread change is worse than VaR, suspend
   current_spread_change = self._spread.diff().iloc[-1]
   if current_spread_change < var * 1.5:  # 1.5x VaR breach
       return {"BTC": 0.0, "ETH": 0.0}
   ```

4. **Z-score**:
   ```python
   X_last = self._spread.iloc[-1]
   z = (X_last - self.mu_ou) / (self.sigma_ou + 1e-8)
   ```

5. **Position sizing**:
   ```
   if z > z_entry:   spread is above mean → short ETH, long BTC
       w_eth = -max_weight
       w_btc = +max_weight * beta   (hedge ratio normalised)
   elif z < -z_entry: spread is below mean → long ETH, short BTC
       w_eth = +max_weight
       w_btc = -max_weight * beta
   elif |z| < z_exit:  near mean → flat
       w_eth = 0.0
       w_btc = 0.0
   else:             in between entry and exit → hold current (simplified: flat)
       w_eth = 0.0
       w_btc = 0.0
   ```
   Clip weights: `w_btc = clip(w_btc, -max_weight, max_weight)`.
   Note: going long/short the *spread* (`X = log ETH - β log BTC`) means:
   - Spread long → buy ETH, sell BTC (in proportion β).
   - Spread short → sell ETH, buy BTC.

6. Return `{"BTC": w_btc, "ETH": w_eth}`.

### Helper: `spread_zscore(prices) -> float`

Returns the current z-score of the spread. Used by the orchestrator for diagnostics.

---

## Risk gates — implementation note

Do NOT import from `core/backtest/`. Implement the drawdown and VaR checks
inline as shown above using `numpy` and the stored spread series. These are
simple rolling computations, not the full metrics module.

---

## Testing

Place tests in `backend/core/tests/strats/mean_reversion/`.

### `test_mean_reversion_strategy.py`

```python
# 1. Cointegrated prices → spread_stationary = True after fit()
# 2. Non-cointegrated (random walk) prices → ADF gate fires, returns zero weights
# 3. Spread above z_entry → ETH weight < 0, BTC weight > 0
# 4. Spread below -z_entry → ETH weight > 0, BTC weight < 0
# 5. predict_signal always returns {"BTC": w1, "ETH": w2}
# 6. Weights always in [-max_weight, max_weight]
# 7. Drawdown gate: after large adverse spread moves, returns zero weights
# 8. MeanReversionStrategy satisfies StrategyProtocol
```

---

## Generating test data

For a **cointegrated** pair (to pass the ADF gate in tests):
```python
import numpy as np
import pandas as pd

def make_cointegrated_prices(n=500, beta=1.5, seed=0):
    rng = np.random.default_rng(seed)
    log_btc = np.cumsum(rng.standard_normal(n) * 0.02)  # BTC random walk
    # OU process for spread
    ou = np.zeros(n)
    kappa, mu, sigma_ou = 0.1, 0.0, 0.05
    for t in range(1, n):
        ou[t] = ou[t-1] + kappa * (mu - ou[t-1]) + sigma_ou * rng.standard_normal()
    log_eth = beta * log_btc + ou  # cointegrated ETH
    idx = pd.date_range("2020-01-01", periods=n, freq="D")
    return pd.DataFrame({
        "BTC": np.exp(log_btc) * 10000,
        "ETH": np.exp(log_eth) * 500,
    }, index=idx)
```

---

## Imports allowed

```python
from core.strats.base import StrategyProtocol
import numpy as np
import pandas as pd
from scipy.stats import linregress
from statsmodels.tsa.stattools import adfuller
```

Do NOT import from `core/backtest/`, `core/evaluation/`, or `core/models/`.

---

## Key empirical constraints

- **Structural breaks** (ETH 2.0 ~2022, BTC ETF ~2024) shift the cointegrating vector.
  Re-estimate `β` at every `fit()` call (walk-forward). Do not hard-code `β`.
- **Half-life gate**: if `half_life > 60` bars the OU reversion is too slow to be
  tradeable — return zero weights even if ADF passes. Add this check in `predict_signal`.
- **Drawdown gate takes priority** over the z-score signal — always check risk gates first.
- The ADF test requires `len(spread) >= 20` to be meaningful; return zero weights if
  the price history is shorter than this.
