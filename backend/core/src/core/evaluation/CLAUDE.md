# CLAUDE.md — `core/evaluation/` subagent briefing

## Role

You are building the **model forecast evaluation harness** for the BTC/ETH regime-switching pipeline.
Your job is to score the predictive outputs of `core/models/` **out-of-sample** against realised values.

**You do NOT run strategies or compute portfolio P&L here.** That is `core/backtest/`.
You evaluate model quality: how accurately do volatility forecasts, price-direction forecasts,
and co-movement estimates match what actually happened?

---

## Module layout to implement

```
core/evaluation/
├── __init__.py
├── CLAUDE.md                  ← this file
├── metrics/
│   ├── __init__.py
│   ├── vol.py                 # QLIKE, MSE, MAE, MZ regression for vol forecasts
│   └── risk.py                # VaR violation tests (Kupiec, Christoffersen)
├── validation/
│   ├── __init__.py
│   └── splitter.py            # walk-forward / expanding-window split generator
└── reports/
    ├── __init__.py
    └── compare.py             # model comparison table, Diebold-Mariano test
```

---

## Standard evaluation contract

**Every model being evaluated must conform to this protocol:**

```python
class ForecastModel(Protocol):
    def fit(self, train: pd.DataFrame) -> None:
        """Fit the model on training data. train is indexed by UTC timestamp."""
        ...

    def predict(self, test: pd.DataFrame) -> pd.Series | pd.DataFrame:
        """
        Return out-of-sample forecasts aligned to test.index.
        - Volatility models: pd.Series of predicted conditional std dev (σ_t).
        - Price/direction models: pd.Series of predicted returns or signals.
        - Co-movement models: pd.DataFrame with columns per pair (e.g. 'BTC_ETH_corr').
        """
        ...
```

Metric functions consume `(forecast, realised)` pairs — both `pd.Series` aligned on the same index.

---

## validation/splitter.py

Implement a **walk-forward splitter** that yields `(train_idx, test_idx)` index pairs:

```python
def walk_forward_splits(
    index: pd.DatetimeIndex,
    min_train: int,          # minimum training bars before first test window
    test_size: int,          # number of bars per test fold
    gap: int = 0,            # bars to skip between train end and test start (avoid leakage)
    expanding: bool = True,  # True = expanding window; False = rolling window of fixed length
) -> Iterator[tuple[pd.DatetimeIndex, pd.DatetimeIndex]]:
    ...
```

This generator is used by all evaluation routines to ensure no data leakage.

---

## metrics/vol.py

Score volatility forecasts `σ̂_t` against a realised volatility proxy `σ_t`
(typically `|r_t|` or a rolling realised std).

| Function | Formula |
|---|---|
| `qlike(forecast, realised)` | `mean(σ_t² / σ̂_t² - log(σ_t² / σ̂_t²) - 1)` — robust loss, scale-invariant |
| `mse_vol(forecast, realised)` | `mean((σ̂_t - σ_t)²)` |
| `mae_vol(forecast, realised)` | `mean(abs(σ̂_t - σ_t))` |
| `mincer_zarnowitz(forecast, realised)` | OLS of `σ_t` on `(1, σ̂_t)`; return `(α, β, R²)` — β≈1, α≈0 for unbiased model |

All functions return a scalar (or a small dict for MZ). They accept `pd.Series` with matching index.

---

## metrics/risk.py

Score **VaR forecasts** (Value-at-Risk) against realised returns.

**Kupiec unconditional coverage test (POF)**:
```
H0: violation rate = α
LR_uc = -2 log L(α) + 2 log L(p_hat)
~ χ²(1) under H0
```

**Christoffersen conditional coverage test**:
```
LR_cc = LR_uc + LR_ind    (independence of violations)
~ χ²(2) under H0
```

```python
def kupiec_test(
    violations: pd.Series,   # boolean series: True where return < -VaR
    alpha: float,            # nominal coverage level (e.g. 0.05)
) -> dict[str, float]:
    """Returns {'lr_stat': ..., 'p_value': ..., 'violation_rate': ...}"""
    ...

def christoffersen_test(
    violations: pd.Series,
    alpha: float,
) -> dict[str, float]:
    """Returns {'lr_stat': ..., 'p_value': ..., 'lr_uc': ..., 'lr_ind': ...}"""
    ...
```

---

## reports/compare.py

**Model comparison table**: given a dict of `{model_name: forecast_series}` and a single
`realised` series, return a `pd.DataFrame` with models as rows and metrics as columns.

```python
def comparison_table(
    forecasts: dict[str, pd.Series],
    realised: pd.Series,
    metrics: list[str] = ["qlike", "mse", "mae"],
) -> pd.DataFrame:
    ...
```

**Diebold-Mariano test**: compare two forecast series on equal predictive accuracy.

```python
def diebold_mariano(
    forecast_a: pd.Series,
    forecast_b: pd.Series,
    realised: pd.Series,
    loss: str = "mse",       # "mse" | "mae" | "qlike"
    h: int = 1,              # forecast horizon (for HAC correction)
) -> dict[str, float]:
    """Returns {'dm_stat': ..., 'p_value': ...}. Positive stat = A is worse than B."""
    ...
```

Use Newey-West (HAC) standard errors for `h > 1`.

---

## Key design constraints (from the research study)

- **Volatility is clustered, heavy-tailed, and regime-switching.** QLIKE is the preferred loss
  for vol models because it is robust to large outliers and is the natural MLE loss for
  GARCH-family models. Always report QLIKE alongside MSE.
- **Lower-tail dependence has strengthened over time.** VaR tests (`risk.py`) must be applied
  specifically to the left tail (losses); test both 95% and 99% VaR levels.
- **Walk-forward splits must enforce a gap of ≥ 1 bar** between train and test to prevent
  any information leakage from the training-set tail.
- **Structural breaks** (ETH 2.0, BTC ETF approvals) mean full-sample evaluation is misleading;
  always evaluate on rolling sub-periods as well as the full out-of-sample window.

---

## Testing

Place tests in `backend/core/tests/evaluation/`. Run with:
```bash
cd backend/core && poetry run pytest tests/evaluation/
```

Minimum test coverage:
- `test_splitter.py` — verify no overlap between train and test, correct fold count,
  expanding vs. rolling window behaviour.
- `test_vol_metrics.py` — unit-test `qlike`, `mse_vol`, `mae_vol` against hand-calculated values;
  test that QLIKE is undefined when `forecast <= 0` (assert raises or returns NaN).
- `test_risk_metrics.py` — synthetic violation series with known rate; verify Kupiec LR stat
  matches the closed-form formula; verify p-value < 0.05 when violation rate is badly mis-specified.
- `test_compare.py` — two toy forecast series, assert `comparison_table` returns correct shape
  and column names; assert `diebold_mariano` returns a dict with `dm_stat` and `p_value`.

---

## Interfaces you depend on (do not modify)

- `core/models/forecast/volatility/` — volatility models; call via `ForecastModel` protocol above.
- `core/models/forecast/price/` — price/direction models; same protocol.
- `core/models/co_mov/` — co-movement models (correlation, tail dependence); same protocol.

If a model class does not yet exist, mock it in tests using `unittest.mock.MagicMock` or a
simple synthetic class that implements the `ForecastModel` protocol.
