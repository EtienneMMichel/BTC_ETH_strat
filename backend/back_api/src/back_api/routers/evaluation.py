from __future__ import annotations

import asyncio
from typing import Any

import pandas as pd
from fastapi import APIRouter, HTTPException, Request

from back_api.jobs import JobStore
from back_api.schemas.evaluation import (
    DMTestRequest,
    EvalJobStatusResponse,
    VarTestRequest,
    VolComparisonRequest,
)

router = APIRouter()


def _get_store(request: Request) -> JobStore:
    return request.app.state.job_store


# ---------------------------------------------------------------------------
# CPU-bound work (runs in thread pool via asyncio.to_thread)
# ---------------------------------------------------------------------------

def _run_vol_comparison_sync(req: VolComparisonRequest) -> dict[str, Any]:
    from core.evaluation.reports.compare import comparison_table

    idx = pd.to_datetime(req.index, utc=True)
    realised = pd.Series(req.realised, index=idx)
    forecasts = {
        name: pd.Series(vals, index=idx)
        for name, vals in req.forecasts.items()
    }
    table = comparison_table(forecasts, realised, req.metrics)
    return table.to_dict(orient="index")


def _run_var_test_sync(req: VarTestRequest) -> dict[str, Any]:
    from core.evaluation.metrics.risk import christoffersen_test, kupiec_test

    idx = pd.to_datetime(req.index, utc=True)
    returns = pd.Series(req.returns, index=idx)
    var_series = pd.Series(req.var_series, index=idx)

    # VaR violation: return < var (var is already negative)
    violations = returns < var_series

    kupiec = kupiec_test(violations, req.alpha)
    christoffersen = christoffersen_test(violations, req.alpha)

    return {"kupiec": kupiec, "christoffersen": christoffersen}


def _run_dm_test_sync(req: DMTestRequest) -> dict[str, Any]:
    from core.evaluation.reports.compare import diebold_mariano

    idx = pd.to_datetime(req.index, utc=True)
    fa = pd.Series(req.forecast_a, index=idx)
    fb = pd.Series(req.forecast_b, index=idx)
    rv = pd.Series(req.realised, index=idx)

    return diebold_mariano(fa, fb, rv, loss=req.loss, h=req.h)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.post("/vol-comparison")
async def vol_comparison(req: VolComparisonRequest, request: Request) -> dict[str, str]:
    """Submit a volatility model comparison job."""
    store: JobStore = _get_store(request)
    job_id = store.create()
    asyncio.create_task(
        store.run(job_id, asyncio.to_thread(_run_vol_comparison_sync, req))
    )
    return {"job_id": job_id}


@router.post("/var-test")
async def var_test(req: VarTestRequest, request: Request) -> dict[str, str]:
    """Submit a VaR backtest (Kupiec + Christoffersen) job."""
    store: JobStore = _get_store(request)
    job_id = store.create()
    asyncio.create_task(
        store.run(job_id, asyncio.to_thread(_run_var_test_sync, req))
    )
    return {"job_id": job_id}


@router.post("/dm-test")
async def dm_test(req: DMTestRequest, request: Request) -> dict[str, str]:
    """Submit a Diebold-Mariano equal predictive accuracy test job."""
    store: JobStore = _get_store(request)
    job_id = store.create()
    asyncio.create_task(
        store.run(job_id, asyncio.to_thread(_run_dm_test_sync, req))
    )
    return {"job_id": job_id}


@router.get("/{job_id}", response_model=EvalJobStatusResponse)
async def get_eval_status(job_id: str, request: Request) -> EvalJobStatusResponse:
    """Poll evaluation job status."""
    store: JobStore = _get_store(request)
    rec = store.get(job_id)
    if rec is None:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found.")
    return EvalJobStatusResponse(
        job_id=job_id,
        status=rec.status,
        result=rec.result if rec.status == "done" else None,
        error=rec.error,
    )
