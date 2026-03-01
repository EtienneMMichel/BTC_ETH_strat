# CLAUDE.md — `core/models/co_mov/` subagent briefing

## Role

You are the **co-movement modelling specialist** for the BTC/ETH portfolio pipeline.
Your responsibility is to estimate and forecast the time-varying **correlation** and
**tail dependence** between BTC and ETH returns. These estimates feed both strategy families
(momentum and mean-reversion) and the regime-switching orchestrator in `core/strats/`.

A `SKILL.md` file will be added here later with deeper domain knowledge (model equations,
calibration procedures, literature references). Until then, use this file as your primary brief.

---

## Module layout to implement

```
core/models/co_mov/
├── __init__.py
├── CLAUDE.md                  ← this file
├── correlation/
│   ├── __init__.py
│   ├── dcc.py                 # DCC-GARCH (Engle 2002) — primary model
│   └── bekk.py                # BEKK-GARCH (Engle & Kroner 1995) — alternative
├── tail/
│   ├── __init__.py
│   ├── copula.py              # Static and time-varying copulas (Gaussian, Student-t, Clayton)
│   └── es_cavar.py            # ES-CAViaR for joint tail risk
└── base.py                    # Abstract base class shared by all co-movement models
```

---

## base.py — shared interface

All co-movement models must implement this protocol so `core/evaluation/` can score them uniformly:

```python
from abc import ABC, abstractmethod
import pandas as pd

class CoMovModel(ABC):
    @abstractmethod
    def fit(self, returns: pd.DataFrame) -> None:
        """
        Fit on a DataFrame of log-returns with columns ['BTC', 'ETH'],
        indexed by UTC timestamp.
        """

    @abstractmethod
    def predict(self, returns: pd.DataFrame) -> pd.DataFrame:
        """
        Return out-of-sample estimates aligned to returns.index.
        Minimum required columns: ['correlation', 'lower_tail_dep'].
        Additional columns allowed (e.g., 'upper_tail_dep', 'cov_BTC_ETH').
        """
```

---

## correlation/dcc.py — DCC-GARCH (primary model)

**Model**: Dynamic Conditional Correlation (Engle 2002).

1. Fit univariate GARCH(1,1) to each return series to obtain standardised residuals `ε_t`.
2. Estimate the DCC parameters `(a, b)` from the quasi-correlation dynamics:
   ```
   Q_t = (1 - a - b) Q̄ + a ε_{t-1} ε_{t-1}' + b Q_{t-1}
   R_t = diag(Q_t)^{-1/2} Q_t diag(Q_t)^{-1/2}
   ```
3. Expose `fit(returns)` and `predict(returns)` per `CoMovModel`.
4. `predict` returns a DataFrame with at least `correlation` = off-diagonal element of `R_t`.

**Key constraint**: correlation must stay in `(-1, 1)` at all times — add a clipping guard.

---

## correlation/bekk.py — BEKK-GARCH (alternative)

**Model**: BEKK(1,1) parametrisation of the conditional covariance matrix `H_t`:
```
H_t = C'C + A' ε_{t-1} ε_{t-1}' A + B' H_{t-1} B
```
- `C` lower triangular, `A` and `B` square 2×2.
- Positive definiteness is guaranteed by construction.
- Expose the same `CoMovModel` interface.
- `predict` returns `correlation` derived from `H_t`, plus `cov_BTC_ETH`.

---

## tail/copula.py — Copula models

Implement three static copulas and one rolling-window version of each:

| Copula | Use case |
|--------|----------|
| Gaussian | Baseline; symmetric tail dependence (none by construction) |
| Student-t | Symmetric heavy tails; parameter `ν` (degrees of freedom) |
| Clayton | Asymmetric; captures **lower-tail dependence** (λ_L > 0) |

For each copula:
- `fit(u, v)` where `u`, `v` are probability-integral-transformed marginals ∈ (0,1).
- `lower_tail_dep()` → scalar λ_L (0 for Gaussian, formula for t and Clayton).
- `upper_tail_dep()` → scalar λ_U.

Rolling estimation: a wrapper `RollingCopula(copula, window)` that re-fits on each
`window`-length rolling slice and returns a time-series of `(lower_tail_dep, upper_tail_dep)`.

---

## tail/es_cavar.py — ES-CAViaR

**Model**: ES-CAViaR jointly estimates VaR and ES for the portfolio return.

Asymmetric slope specification:
```
VaR_t(α) = β_0 + β_1 VaR_{t-1} + β_2 r_{t-1}^- + β_3 r_{t-1}^+
ES_t(α)  = VaR_t(α) / α  (simplified strict proportionality version)
```

- Fit via quantile regression (minimise pinball loss at level `α`).
- Expose `CoMovModel` interface; `predict` returns columns `['var', 'es']`.

---

## Key empirical constraints (from the research study)

- **Correlation is positive and time-varying**, spiking in crisis — use DCC as the default;
  BEKK as robustness check.
- **Lower-tail dependence has strengthened over time** — Clayton copula is the preferred
  tail model; always test whether `λ_L > 0` significantly.
- **Student-t copula is preferred over Gaussian** for any joint risk calculation because
  of heavy tails and simultaneous crash risk.
- Structural breaks (ETH 2.0 ~2022, BTC ETF ~2024) may shift correlation regimes — the
  rolling copula wrapper exists precisely for this.

---

## Testing

Place tests in `backend/core/tests/models/co_mov/`. Minimum coverage:

- `test_dcc.py` — synthetic bivariate normal returns, fit DCC, assert `correlation` ∈ (-1,1)
  at all time steps, shape matches input.
- `test_bekk.py` — same synthetic data, assert positive-definite `H_t` at each step.
- `test_copula.py` — perfect correlated uniform sample → Clayton lower tail dep ≈ 1;
  independent sample → ≈ 0.
- `test_es_cavar.py` — assert VaR series is always negative (left tail); ES ≤ VaR.

---

## Interfaces you expose (consumers)

- `core/strats/` — uses `correlation` and `lower_tail_dep` from `predict()` as regime signals.
- `core/evaluation/` — scores your `predict()` output against realised correlation proxies.
- Do **not** import from `core/strats/` or `core/backtest/` — dependency flows downward only.
