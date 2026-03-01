# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Goal

A regime-switching pipeline for a BTC/ETH two-asset portfolio. A master orchestrator detects the current market regime and routes to the best-fit sub-strategy. Two strategy families are in scope:

- **Momentum** — time-series momentum / trend-following on BTC and ETH individually
- **Mean Reversion** — relative-value on the BTC-ETH spread (Ornstein-Uhlenbeck formulation); regime health measured via drawdown and VaR

## pipeline
  evaluation/ ──scores──▶ models/ ──signals──▶ strats/ ──weights──▶ backtest/
                            │                    │
                    vol/price/co_mov     momentum/mean_rev/orchestrator

## Commands

```bash
cd backend/core

# Install dependencies
python -m poetry install

# Run all tests
python -m poetry run pytest

# Run a single test file
python -m poetry run pytest tests/path/to/test_file.py

# Run a single test by name
python -m poetry run pytest tests/path/to/test_file.py::test_function_name
```

## Architecture

```
backend/core/
├── pyproject.toml          # Poetry project, Python >=3.13, src/ layout
└── src/core/
    ├── models/
    │   ├── co_mov/         # BTC-ETH co-movement: correlation, DCC-GARCH, tail-dependence
    │   └── forecast/
    │       ├── price/      # Price forecasting (momentum signals, trend)
    │       └── volatility/ # Vol forecasting (GARCH variants, realized vol, EWMA)
    └── strats/             # Strategy implementations + regime-switching orchestrator
```

### Design Principles from the Research Study (`study/resume.pdf`)

**Regime-switching orchestrator** (`strats/`): classifies the market into regimes (e.g., low-vol/trending vs. high-vol/mean-reverting) using a Markov-switching or threshold framework, then dispatches to the appropriate sub-strategy. Key signal sources: volatility level, drawdown, VaR/ES.

**Momentum strategy**: activated in trending/low-vol regimes. Signal: `sign(Σ r_{i,t-k})` with vol-scaling. Requires `forecast/price/` and `forecast/volatility/`.

**Mean-reversion strategy**: activated when spread is stationary and regime assumptions hold. Models the log-price spread `X_t = log P_ETH - β log P_BTC` as an OU process `dX_t = κ(μ - X_t)dt + σ dW_t`. Risk measured by drawdown and VaR. Requires `co_mov/` and `forecast/volatility/`.

**Co-movement module** (`models/co_mov/`): estimates time-varying BTC-ETH correlation and tail dependence (VAR-AGARCH, BEKK, or DCC). Feeds both strategy families and the regime classifier.

### Key Empirical Facts (from `study/resume.pdf`) to Respect in All Models

- BTC-ETH correlation is **positive and time-varying**, rising sharply in crisis regimes — diversification shrinks exactly when needed most.
- Volatility is **clustered, heavy-tailed, and regime-switching** — always use GARCH-family or realized-vol models, never constant-vol assumptions.
- **Lower-tail dependence has strengthened** over time — joint crash risk must be modeled explicitly (copulas or ES-CAViaR).
- BTC dominates optimal weights due to lower relative risk; the spread `X_t` is subject to structural breaks (ETH 2.0, ETF approvals).
- Volatility-scaling prevents procyclical deleveraging; threshold/regime-switching corrections are preferred over plain GARCH in tail-risk contexts.
