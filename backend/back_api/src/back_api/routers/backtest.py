from __future__ import annotations

import asyncio
import json
import threading
from typing import Any

import pandas as pd
from fastapi import APIRouter, HTTPException, Request

from back_api.config import get_settings
from back_api.jobs import JobStore
from back_api.schemas.backtest import (
    BacktestRequest,
    BacktestResultSchema,
    JobStatusResponse,
)

router = APIRouter()

# ---------------------------------------------------------------------------
# Layer 1 — Data loading cache (threading.Lock: runs in thread pool)
# ---------------------------------------------------------------------------

_DATA_CACHE: dict[tuple, pd.DataFrame] = {}
_DATA_CACHE_LOCK = threading.Lock()
_DATA_CACHE_MAX = 32  # ~60 KB/frame × 32 ≈ 2 MB


def _data_cache_key(data_dir: str, assets: dict[str, str], freq: str) -> tuple:
    return (data_dir, tuple(sorted(assets.items())), freq)


# ---------------------------------------------------------------------------
# Layer 2 — Backtest result cache (threading.Lock: runs in thread pool)
# ---------------------------------------------------------------------------

_RESULT_CACHE: dict[str, BacktestResultSchema] = {}
_RESULT_CACHE_LOCK = threading.Lock()
_RESULT_CACHE_MAX = 128  # ~200 KB/result × 128 ≈ 25 MB


def _request_fingerprint(req: BacktestRequest) -> str:
    """Stable canonical JSON string of all request fields."""
    d = req.model_dump()
    d["assets"] = dict(sorted(d["assets"].items()))
    return json.dumps(d, sort_keys=True)


def _get_store(request: Request) -> JobStore:
    return request.app.state.job_store


# ---------------------------------------------------------------------------
# CPU-bound work (runs in thread pool via asyncio.to_thread)
# ---------------------------------------------------------------------------

def _run_backtest_sync(req: BacktestRequest) -> BacktestResultSchema:
    """Synchronous backtest logic — called via asyncio.to_thread."""
    from core.backtest.data.loader import load_assets
    from core.backtest.engine.runner import WalkForwardRunner

    # Layer 2: check result cache before any expensive work
    fingerprint = _request_fingerprint(req)
    with _RESULT_CACHE_LOCK:
        cached = _RESULT_CACHE.get(fingerprint)
    if cached is not None:
        return cached

    settings = get_settings()
    data_dir = req.data_dir if req.data_dir else str(settings.data_dir)

    # Layer 1: check data cache before disk I/O
    dk = _data_cache_key(data_dir, req.assets, req.freq)
    with _DATA_CACHE_LOCK:
        ohlcv = _DATA_CACHE.get(dk)
    if ohlcv is None:
        ohlcv = load_assets(data_dir, req.assets, req.freq)
        with _DATA_CACHE_LOCK:
            _DATA_CACHE[dk] = ohlcv
            while len(_DATA_CACHE) > _DATA_CACHE_MAX:
                del _DATA_CACHE[next(iter(_DATA_CACHE))]

    # Extract close prices
    prices = pd.DataFrame(
        {name: ohlcv[name]["close"] for name in req.assets},
    )

    # Instantiate strategy
    if req.strategy == "momentum":
        from core.strats.momentum.strategy import MomentumStrategy
        strategy = MomentumStrategy()
    elif req.strategy == "mean_reversion":
        from core.strats.mean_reversion.strategy import MeanReversionStrategy
        strategy = MeanReversionStrategy()
    else:
        from core.strats.orchestrator.regime import OrchestratorStrategy
        strategy = OrchestratorStrategy()

    runner = WalkForwardRunner(
        strategy=strategy,
        min_train_bars=req.min_train_bars,
        rebalance_every=req.rebalance_every,
        fee_rate=req.fee_rate,
        slippage=req.slippage,
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

    trade_count = len(result.trade_log)

    schema = BacktestResultSchema(
        equity_curve=equity_curve,
        metrics={k: float(v) for k, v in result.metrics.items()},
        trade_count=trade_count,
        signals=signals,
    )

    # Layer 2: store result in cache
    with _RESULT_CACHE_LOCK:
        _RESULT_CACHE[fingerprint] = schema
        while len(_RESULT_CACHE) > _RESULT_CACHE_MAX:
            del _RESULT_CACHE[next(iter(_RESULT_CACHE))]

    return schema


async def _backtest_coro(req: BacktestRequest) -> BacktestResultSchema:
    return await asyncio.to_thread(_run_backtest_sync, req)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.post("/run")
async def run_backtest(req: BacktestRequest, request: Request) -> dict[str, str]:
    """Submit a backtest job. Returns immediately with a job_id."""
    store: JobStore = _get_store(request)
    # Layer 3: deduplicate identical in-flight / completed jobs
    fingerprint = _request_fingerprint(req)
    job_id, is_new = await store.find_or_create(fingerprint)
    if is_new:
        asyncio.create_task(store.run(job_id, _backtest_coro(req)))
    return {"job_id": job_id}


@router.get("/{job_id}", response_model=JobStatusResponse)
async def get_backtest_status(job_id: str, request: Request) -> JobStatusResponse:
    """Poll backtest job status."""
    store: JobStore = _get_store(request)
    rec = store.get(job_id)
    if rec is None:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found.")
    return JobStatusResponse(
        job_id=job_id,
        status=rec.status,
        result=rec.result if rec.status == "done" else None,
        error=rec.error,
    )
