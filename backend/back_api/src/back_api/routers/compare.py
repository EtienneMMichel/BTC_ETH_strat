from __future__ import annotations

import asyncio
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException, Request

from back_api.config import get_settings
from back_api.jobs import JobStore
from back_api.schemas.backtest import BacktestRequest, MarkowitzParams
from back_api.schemas.compare import (
    CompareJobResponse,
    CompareRequest,
    CompareResult,
    FrontierPoint,
    StrategyConfig,
    StrategyResultEntry,
)

router = APIRouter()

# ---------------------------------------------------------------------------
# Data loading cache (shared with backtest router pattern)
# ---------------------------------------------------------------------------

_DATA_CACHE: dict[tuple, pd.DataFrame] = {}
_DATA_CACHE_LOCK = threading.Lock()
_DATA_CACHE_MAX = 32


def _data_cache_key(data_dir: str, assets: dict[str, str], freq: str) -> tuple:
    return (data_dir, tuple(sorted(assets.items())), freq)


def _get_store(request: Request) -> JobStore:
    return request.app.state.job_store


# ---------------------------------------------------------------------------
# Strategy instantiation (mirrors backtest router)
# ---------------------------------------------------------------------------


def _make_strategy(cfg: StrategyConfig):
    """Instantiate a strategy from config. Lazy imports."""
    if cfg.strategy == "momentum":
        from core.strats.momentum.strategy import MomentumStrategy
        return MomentumStrategy()
    elif cfg.strategy == "mean_reversion":
        from core.strats.mean_reversion.strategy import MeanReversionStrategy
        return MeanReversionStrategy()
    elif cfg.strategy == "markowitz":
        from core.strats.markowitz.strategy import MarkowitzStrategy
        from core.models.covariance.rolling import RollingCovModel
        from core.models.covariance.diagonal import DiagonalCovModel
        from core.models.covariance.bekk_cov import BEKKCovModel
        from core.models.expected_returns.rolling_mean import RollingMeanReturns
        from core.models.expected_returns.signal import SignalExpectedReturns
        from core.models.forecast.price.momentum import TSMOMModel

        p = cfg.markowitz
        cov_map = {
            "rolling": RollingCovModel,
            "diagonal": DiagonalCovModel,
            "bekk": BEKKCovModel,
        }
        er_map: dict[str, Any] = {
            "rolling_mean": lambda: RollingMeanReturns(),
            "signal_tsmom": lambda: SignalExpectedReturns(model_cls=TSMOMModel),
        }
        return MarkowitzStrategy(
            cov_model=cov_map[p.cov_model](),
            expected_returns=er_map[p.er_model](),
            objective=p.objective,
            gamma=p.gamma,
            long_only=p.long_only,
            max_weight=p.max_weight,
            min_weight=p.min_weight,
            risk_free_rate=p.risk_free_rate,
            target_vol=p.target_vol,
            min_history=p.min_history,
        )
    else:
        from core.strats.orchestrator.regime import OrchestratorStrategy
        return OrchestratorStrategy()


# ---------------------------------------------------------------------------
# Run a single strategy backtest
# ---------------------------------------------------------------------------


def _run_single(
    prices: pd.DataFrame,
    cfg: StrategyConfig,
    idx: int,
    min_train_bars: int,
    rebalance_every: int,
    fee_rate: float,
    slippage: float,
    assets: dict[str, str],
) -> StrategyResultEntry:
    from core.backtest.engine.runner import WalkForwardRunner

    strategy = _make_strategy(cfg)
    runner = WalkForwardRunner(
        strategy=strategy,
        min_train_bars=min_train_bars,
        rebalance_every=rebalance_every,
        fee_rate=fee_rate,
        slippage=slippage,
    )
    result = runner.run(prices)

    # Serialize equity curve
    equity_curve = [
        (ts.isoformat(), float(val))
        for ts, val in result.equity_curve.items()
    ]

    # Serialize signals
    if result.signals.empty:
        signals: list[tuple[str, dict[str, float]]] = []
    else:
        signals = [
            (
                ts.isoformat(),
                {
                    col.removeprefix("w_"): float(result.signals.loc[ts, col])
                    for col in result.signals.columns
                },
            )
            for ts in result.signals.index
        ]

    # Annualised metrics
    eq = result.equity_curve
    n_bars = len(eq)
    ann_return = float((eq.iloc[-1] / eq.iloc[0]) ** (252 / max(n_bars, 1)) - 1)
    daily_rets = eq.pct_change().dropna()
    ann_vol = float(daily_rets.std() * np.sqrt(252))

    label = cfg.label if cfg.label else f"{cfg.strategy}_{idx}"

    return StrategyResultEntry(
        label=label,
        strategy=cfg.strategy,
        equity_curve=equity_curve,
        metrics={k: float(v) for k, v in result.metrics.items()},
        trade_count=len(result.trade_log),
        signals=signals,
        ann_return=ann_return,
        ann_vol=ann_vol,
    )


# ---------------------------------------------------------------------------
# Efficient frontier computation
# ---------------------------------------------------------------------------


def _compute_frontier(
    prices: pd.DataFrame,
    req: CompareRequest,
) -> tuple[list[FrontierPoint], dict[str, FrontierPoint]]:
    from core.models.covariance.rolling import RollingCovModel
    from core.models.covariance.diagonal import DiagonalCovModel
    from core.models.covariance.bekk_cov import BEKKCovModel
    from core.models.expected_returns.rolling_mean import RollingMeanReturns
    from core.models.expected_returns.signal import SignalExpectedReturns
    from core.models.forecast.price.momentum import TSMOMModel
    from core.strats.markowitz.optimizer import (
        mean_variance_weights,
        min_variance_weights,
        max_sharpe_weights,
    )

    cov_map = {
        "rolling": RollingCovModel,
        "diagonal": DiagonalCovModel,
        "bekk": BEKKCovModel,
    }
    er_map_cls: dict[str, Any] = {
        "rolling_mean": lambda: RollingMeanReturns(),
        "signal_tsmom": lambda: SignalExpectedReturns(model_cls=TSMOMModel),
    }

    log_returns = np.log(prices / prices.shift(1)).dropna()

    # Fit covariance model
    cov_model = cov_map[req.frontier_cov_model]()
    cov_model.fit(log_returns)
    cov_df = cov_model.predict(log_returns)

    # Fit ER model
    er_model = er_map_cls[req.frontier_er_model]()
    er_model.fit(prices)
    er_df = er_model.predict(prices)

    # Extract last-bar estimates
    last_cov = cov_df.iloc[-1]
    sigma_matrix = np.array([
        [last_cov["var_BTC"], last_cov["cov_BTC_ETH"]],
        [last_cov["cov_BTC_ETH"], last_cov["var_ETH"]],
    ])
    mu = np.array([er_df["BTC"].iloc[-1], er_df["ETH"].iloc[-1]])

    asset_names = list(prices.columns)
    lo = req.frontier_long_only

    def _make_point(gamma: float, w: np.ndarray) -> FrontierPoint:
        port_ret = float(w @ mu) * 252
        port_vol = float(np.sqrt(max(w @ sigma_matrix @ w, 1e-16))) * np.sqrt(252)
        sharpe = port_ret / (port_vol + 1e-12)
        return FrontierPoint(
            gamma=gamma,
            ann_return=port_ret,
            ann_vol=port_vol,
            sharpe=sharpe,
            weights={asset_names[i]: float(w[i]) for i in range(len(asset_names))},
        )

    # Sweep gamma for mean-variance frontier
    gamma_range = np.logspace(-1, 3, req.frontier_points)
    frontier: list[FrontierPoint] = []
    for g in gamma_range:
        w = mean_variance_weights(
            mu, sigma_matrix, gamma=float(g),
            long_only=lo, max_w=1.0, min_w=-1.0 if not lo else 0.0,
        )
        frontier.append(_make_point(float(g), w))

    # Special points
    w_min = min_variance_weights(sigma_matrix, long_only=lo)
    w_max_sr = max_sharpe_weights(mu, sigma_matrix, long_only=lo)

    special = {
        "min_var": _make_point(0.0, w_min),
        "max_sharpe": _make_point(0.0, w_max_sr),
    }

    return frontier, special


# ---------------------------------------------------------------------------
# Main sync worker
# ---------------------------------------------------------------------------


def _run_compare_sync(req: CompareRequest) -> CompareResult:
    from core.backtest.data.loader import load_assets

    settings = get_settings()
    data_dir = req.data_dir if req.data_dir else str(settings.data_dir)

    # Load data (with cache)
    dk = _data_cache_key(data_dir, req.assets, req.freq)
    with _DATA_CACHE_LOCK:
        ohlcv = _DATA_CACHE.get(dk)
    if ohlcv is None:
        ohlcv = load_assets(data_dir, req.assets, req.freq)
        with _DATA_CACHE_LOCK:
            _DATA_CACHE[dk] = ohlcv
            while len(_DATA_CACHE) > _DATA_CACHE_MAX:
                del _DATA_CACHE[next(iter(_DATA_CACHE))]

    prices = pd.DataFrame(
        {name: ohlcv[name]["close"] for name in req.assets},
    )

    # Run strategies in parallel via thread pool
    results: list[StrategyResultEntry] = [None] * len(req.strategies)  # type: ignore[list-item]
    errors: list[str | None] = [None] * len(req.strategies)

    def _worker(idx: int) -> None:
        try:
            results[idx] = _run_single(
                prices, req.strategies[idx], idx,
                req.min_train_bars, req.rebalance_every,
                req.fee_rate, req.slippage, req.assets,
            )
        except Exception as exc:
            errors[idx] = f"{req.strategies[idx].strategy}: {exc}"

    max_workers = min(len(req.strategies), 4)
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = [pool.submit(_worker, i) for i in range(len(req.strategies))]
        for f in futures:
            f.result()  # wait for all

    # Collect errors
    err_msgs = [e for e in errors if e is not None]
    if err_msgs and all(e is not None for e in errors):
        raise RuntimeError("All strategies failed: " + "; ".join(err_msgs))

    strategy_results = [r for r in results if r is not None]

    # ── Align all curves to the latest start date & renormalize to 1.0 ──
    if strategy_results:
        # Find the latest start timestamp across all strategies
        start_dates = [
            pd.Timestamp(s.equity_curve[0][0]) for s in strategy_results if s.equity_curve
        ]
        common_start = max(start_dates) if start_dates else None

        if common_start is not None:
            for i, sr in enumerate(strategy_results):
                # Filter to points >= common_start
                filtered = [(t, v) for t, v in sr.equity_curve if pd.Timestamp(t) >= common_start]
                if not filtered:
                    continue
                # Renormalize so first point = 1.0
                base = filtered[0][1]
                if base != 0:
                    filtered = [(t, v / base) for t, v in filtered]
                # Recompute ann metrics on trimmed curve
                n_bars = len(filtered)
                if n_bars > 1:
                    ann_ret = (filtered[-1][1] / filtered[0][1]) ** (252 / max(n_bars, 1)) - 1
                    daily_rets = pd.Series([filtered[j][1] / filtered[j - 1][1] - 1 for j in range(1, n_bars)])
                    ann_v = float(daily_rets.std() * np.sqrt(252))
                else:
                    ann_ret = 0.0
                    ann_v = 0.0
                strategy_results[i] = StrategyResultEntry(
                    label=sr.label,
                    strategy=sr.strategy,
                    equity_curve=filtered,
                    metrics=sr.metrics,
                    trade_count=sr.trade_count,
                    signals=[(t, w) for t, w in sr.signals if pd.Timestamp(t) >= common_start],
                    ann_return=ann_ret,
                    ann_vol=ann_v,
                )

    # Buy-and-hold benchmarks aligned to same common start
    if strategy_results and strategy_results[0].equity_curve:
        eq_idx = pd.DatetimeIndex([
            pd.Timestamp(t) for t, _ in strategy_results[0].equity_curve
        ])
    else:
        eq_idx = prices.index

    benchmarks: dict[str, list[tuple[str, float]]] = {}
    for asset in req.assets:
        asset_prices = prices[asset].reindex(eq_idx).ffill()
        cumulative = asset_prices / asset_prices.iloc[0]
        benchmarks[asset] = [
            (ts.isoformat(), float(val))
            for ts, val in cumulative.items()
        ]

    # Efficient frontier
    frontier = None
    frontier_special = None
    if req.include_frontier:
        try:
            frontier, frontier_special = _compute_frontier(prices, req)
        except Exception:
            pass  # Frontier is optional; don't fail the whole job

    return CompareResult(
        strategies=strategy_results,
        benchmarks=benchmarks,
        frontier=frontier,
        frontier_special=frontier_special,
    )


async def _compare_coro(req: CompareRequest) -> CompareResult:
    return await asyncio.to_thread(_run_compare_sync, req)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.post("/run")
async def run_compare(req: CompareRequest, request: Request) -> dict[str, str]:
    store: JobStore = _get_store(request)
    job_id = store.create()
    asyncio.create_task(store.run(job_id, _compare_coro(req)))
    return {"job_id": job_id}


@router.get("/{job_id}", response_model=CompareJobResponse)
async def get_compare_status(job_id: str, request: Request) -> CompareJobResponse:
    store: JobStore = _get_store(request)
    rec = store.get(job_id)
    if rec is None:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found.")
    return CompareJobResponse(
        job_id=job_id,
        status=rec.status,
        result=rec.result if rec.status == "done" else None,
        error=rec.error,
    )
