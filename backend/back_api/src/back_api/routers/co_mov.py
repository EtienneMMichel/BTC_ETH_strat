from __future__ import annotations

import asyncio
from typing import Any

import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException, Request

from back_api.config import get_settings
from back_api.jobs import JobStore
from back_api.schemas.co_mov_req import CoMovRequest
from back_api.schemas.evaluation import EvalJobStatusResponse

router = APIRouter()


def _get_store(request: Request) -> JobStore:
    return request.app.state.job_store


def _series_to_timeseries(s: pd.Series) -> list[list]:
    return [
        [ts.isoformat(), float(v)]
        for ts, v in s.dropna().items()
    ]


def _rank_to_uniform(series: pd.Series) -> pd.Series:
    """Probability integral transform: map to uniform marginals via rank."""
    from scipy.stats import rankdata
    n = len(series)
    u = rankdata(series.values, method="average") / (n + 1)
    return pd.Series(u, index=series.index)


def _run_co_mov_sync(req: CoMovRequest) -> dict[str, Any]:
    from core.backtest.data.loader import load_assets
    from core.models.co_mov.correlation.dcc import DCCModel
    from core.models.co_mov.tail.copula import (
        ClaytonCopula,
        GaussianCopula,
        RollingCopula,
        StudentTCopula,
    )

    settings = get_settings()
    data_dir = req.data_dir if req.data_dir else str(settings.data_dir)

    ohlcv_multi = load_assets(data_dir, req.assets, req.freq)

    # Build log-return DataFrame for BTC and ETH
    asset_names = list(req.assets.keys())  # e.g. ["BTC", "ETH"]
    returns = pd.DataFrame(
        {name: np.log(ohlcv_multi[name]["close"]).diff() for name in asset_names}
    ).dropna()

    # ── DCC model ─────────────────────────────────────────────────────────
    dcc = DCCModel()
    dcc.fit(returns)
    dcc_out = dcc.predict(returns)

    dcc_correlation = dcc_out["correlation"]
    dcc_lower_tail = dcc_out["lower_tail_dep"]

    # ── Rolling copula ────────────────────────────────────────────────────
    copula_map = {
        "gaussian": GaussianCopula,
        "student_t": StudentTCopula,
        "clayton": ClaytonCopula,
    }
    copula_cls = copula_map[req.copula_type]

    # Rank-transform each series to uniform marginals
    u = _rank_to_uniform(returns.iloc[:, 0])
    v = _rank_to_uniform(returns.iloc[:, 1])

    rolling_cop = RollingCopula(copula_cls, window=req.rolling_window)
    cop_out = rolling_cop.fit_predict(u, v)

    copula_lower = cop_out["lower_tail_dep"]
    copula_upper = cop_out["upper_tail_dep"]

    # ── Current stats (last non-NaN value) ────────────────────────────────
    def last_valid(s: pd.Series) -> float:
        v = s.dropna()
        return float(v.iloc[-1]) if len(v) > 0 else float("nan")

    current_stats = {
        "dcc_correlation": last_valid(dcc_correlation),
        "dcc_lower_tail_dep": last_valid(dcc_lower_tail),
        "copula_lower_tail_dep": last_valid(copula_lower),
    }

    return {
        "dcc_correlation": _series_to_timeseries(dcc_correlation),
        "dcc_lower_tail_dep": _series_to_timeseries(dcc_lower_tail),
        "copula_lower_tail_dep": _series_to_timeseries(copula_lower),
        "copula_upper_tail_dep": _series_to_timeseries(copula_upper),
        "current_stats": current_stats,
        "freq": req.freq,
        "rolling_window": req.rolling_window,
        "copula_type": req.copula_type,
    }


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.post("/run")
async def run_co_mov(req: CoMovRequest, request: Request) -> dict[str, str]:
    """Submit a co-movement analysis job."""
    store: JobStore = _get_store(request)
    job_id = store.create()
    asyncio.create_task(
        store.run(job_id, asyncio.to_thread(_run_co_mov_sync, req))
    )
    return {"job_id": job_id}


@router.get("/{job_id}", response_model=EvalJobStatusResponse)
async def get_co_mov_status(job_id: str, request: Request) -> EvalJobStatusResponse:
    """Poll co-movement job status."""
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
