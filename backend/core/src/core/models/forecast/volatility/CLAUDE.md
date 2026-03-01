# CLAUDE.md — `core/models/forecast/volatility/` subagent briefing

## Role

You are the **volatility forecasting specialist** for the BTC/ETH portfolio pipeline.
Your responsibility is to model and forecast the conditional volatility (σ_t) of BTC and ETH
individually. These forecasts are consumed by both strategy families for position sizing
(vol-scaling) and by the regime classifier for state detection.

A `SKILL.md` file will be added here later with deeper domain knowledge (likelihood derivations,
parameter constraints, numerical recipes). Until then, use this file as your primary brief.

---

## Module layout to implement

```
core/models/forecast/volatility/
├── __init__.py
├── CLAUDE.md                  ← this file
├── base.py                    # Abstract base class for all vol models
├── garch.py                   # GARCH(p,q), GJR-GARCH (asymmetric), EGARCH
├── realized.py                # Realized volatility (Rogers-Satchell, Yang-Zhang)
└── ewma.py                    # EWMA (RiskMetrics λ-decay)
```

---

## base.py — shared interface

All volatility models must implement this protocol so `core/evaluation/` can score them uniformly:

```python
from abc import ABC, abstractmethod
import pandas as pd

class VolatilityModel(ABC):
    @abstractmethod
    def fit(self, returns: pd.Series) -> None:
        """
        Fit on a pd.Series of log-returns indexed by UTC timestamp.
        Asset identifier is not part of the interface — instantiate one model per asset.
        """

    @abstractmethod
    def predict(self, returns: pd.Series) -> pd.Series:
        """
        Return out-of-sample conditional std-dev forecasts σ_t > 0,
        aligned to returns.index.
        One-step-ahead: predict()[t] uses only information up to t-1.
        """
```

---

## garch.py — GARCH family

### GARCH(p,q)

Standard symmetric specification:
```
σ²_t = ω + Σ_{i=1}^{q} α_i ε²_{t-i} + Σ_{j=1}^{p} β_j σ²_{t-j}
```
Constraints: `ω > 0`, `α_i ≥ 0`, `β_j ≥ 0`, `Σα + Σβ < 1` (stationarity).

### GJR-GARCH (Glosten-Jagannathan-Runkle)

Adds asymmetric leverage term:
```
σ²_t = ω + (α + γ · 𝟙[ε_{t-1} < 0]) ε²_{t-1} + β σ²_{t-1}
```
`γ > 0` captures the leverage effect (negative shocks increase vol more than positive).

### EGARCH (Nelson 1991)

Log-specification — ensures σ²_t > 0 without inequality constraints:
```
log σ²_t = ω + β log σ²_{t-1} + α (|z_{t-1}| - E|z|) + γ z_{t-1}
```
where `z_t = ε_t / σ_t`.

**Implementation notes**:
- Fit all three via maximum likelihood (assume normal or Student-t innovations).
- Expose `fit(returns)` / `predict(returns)` per `VolatilityModel`.
- `predict` returns σ_t (std dev, not variance).
- Expose `aic` and `bic` properties after fitting for model selection.

---

## realized.py — Realized volatility estimators

High-frequency or OHLCV-based non-parametric estimators.

### Rogers-Satchell (RS) estimator

Uses open/high/low/close prices — **unbiased under drift**:
```
RS_t = (h_t - c_t)(h_t - o_t) + (l_t - c_t)(l_t - o_t)
```
where all prices are in log-space.

### Yang-Zhang (YZ) estimator

Extends RS to handle overnight jumps:
```
σ²_YZ = σ²_o + k σ²_RS + (1-k) σ²_c
```
where `σ²_o` = overnight variance, `σ²_c` = close-to-close, `k` = optimal weight.

**Implementation notes**:
- Both estimators accept an OHLCV `pd.DataFrame` with columns `open, high, low, close`.
- Return a `pd.Series` of daily σ_t (annualisation is optional, controlled by a flag).
- These are **in-sample** estimators; they serve as the realised vol proxy for `core/evaluation/`.

---

## ewma.py — EWMA (RiskMetrics)

Simple exponentially weighted moving average:
```
σ²_t = λ σ²_{t-1} + (1-λ) ε²_{t-1}
```
Default `λ = 0.94` (RiskMetrics daily). Expose `fit(returns)` / `predict(returns)` per base class.

EWMA is the cheapest baseline; it should be outperformed by GJR-GARCH in backtests due to
the leverage effect in crypto.

---

## Key empirical constraints (from the research study)

- **Volatility is clustered, heavy-tailed, and regime-switching.** Always use GJR-GARCH or
  EGARCH as the primary models — plain GARCH understates downside vol.
- **Never use constant-vol assumptions** — even EWMA is better than a scalar vol estimate.
- **Leverage effect is present in crypto** (`γ > 0` in GJR-GARCH) — verify this empirically
  in tests using synthetic data with asymmetric shocks.
- **QLIKE loss** is the preferred evaluation criterion for vol forecasts (scale-invariant,
  MLE-consistent) — implemented in `core/evaluation/metrics/vol.py`.

---

## Testing

Place tests in `backend/core/tests/models/forecast/volatility/`. Minimum coverage:

- `test_garch.py` — synthetic GARCH(1,1) process; fit GARCH and GJR-GARCH; assert
  `predict()` returns positive series of correct length; `aic < inf`.
- `test_realized.py` — synthetic OHLCV with known RS formula inputs; assert scalar output
  matches closed-form; assert YZ ≥ 0 always.
- `test_ewma.py` — constant-variance white noise; assert EWMA converges to true σ; assert
  `predict()` shape matches input.
- `test_base_contract.py` — any model that implements `VolatilityModel` must pass:
  `predict(test).index == test.index`, all values > 0.

---

## Interfaces you expose (consumers)

- `core/strats/` — uses `predict()` output as the vol-scaling denominator for position sizing.
- `core/models/co_mov/` — marginal vol estimates feed the DCC-GARCH standardisation step.
- `core/evaluation/metrics/vol.py` — scores your `predict()` against `realized.py` estimates.
- Do **not** import from `core/strats/`, `core/backtest/`, or `core/evaluation/`.
