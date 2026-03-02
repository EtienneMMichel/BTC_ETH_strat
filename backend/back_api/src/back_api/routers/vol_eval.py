from __future__ import annotations

import asyncio
from typing import Any

import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException, Request

from back_api.config import get_settings
from back_api.jobs import JobStore
from back_api.schemas.evaluation import EvalJobStatusResponse
from back_api.schemas.vol_eval import VolEvalRequest

router = APIRouter()


def _get_store(request: Request) -> JobStore:
    return request.app.state.job_store


def _series_to_timeseries(s: pd.Series) -> list[list]:
    """Convert a pd.Series with DatetimeIndex to [[iso_str, float], ...]."""
    return [
        [ts.isoformat(), float(v)]
        for ts, v in s.dropna().items()
    ]


def _run_vol_eval_sync(req: VolEvalRequest) -> dict[str, Any]:
    from core.backtest.data.loader import load_assets
    from core.evaluation.reports.compare import comparison_table
    from core.models.forecast.volatility.realized import rogers_satchell, yang_zhang

    settings = get_settings()
    data_dir = req.data_dir if req.data_dir else str(settings.data_dir)

    ohlcv_multi = load_assets(data_dir, req.assets, req.freq)
    asset = req.asset
    ohlcv = pd.DataFrame({
        col: ohlcv_multi[asset][col]
        for col in ["open", "high", "low", "close", "volume"]
        if col in ohlcv_multi[asset].columns
    })

    close = ohlcv["close"]
    returns = np.log(close).diff().dropna()

    forecasts: dict[str, pd.Series] = {}
    requested = req.models

    # ── Parametric models ──────────────────────────────────────────────────
    if "garch" in requested:
        from core.models.forecast.volatility.garch import GARCHModel
        m = GARCHModel()
        m.fit(returns)
        forecasts["garch"] = m.predict(returns)

    if "gjr_garch" in requested:
        from core.models.forecast.volatility.garch import GJRGARCHModel
        m = GJRGARCHModel()
        m.fit(returns)
        forecasts["gjr_garch"] = m.predict(returns)

    if "egarch" in requested:
        from core.models.forecast.volatility.garch import EGARCHModel
        m = EGARCHModel()
        m.fit(returns)
        forecasts["egarch"] = m.predict(returns)

    if "ewma" in requested:
        from core.models.forecast.volatility.ewma import EWMAModel
        m = EWMAModel()
        m.fit(returns)
        forecasts["ewma"] = m.predict(returns)

    # ── Realized vol models ────────────────────────────────────────────────
    if "rogers_satchell" in requested:
        forecasts["rogers_satchell"] = rogers_satchell(ohlcv)

    if "yang_zhang" in requested:
        forecasts["yang_zhang"] = yang_zhang(ohlcv)

    # ── Realised benchmark: Yang-Zhang (always computed) ──────────────────
    realised = yang_zhang(ohlcv)

    # Align all series on a common index (drop burn-in NaNs)
    all_series = list(forecasts.values()) + [realised]
    common_idx = all_series[0].dropna().index
    for s in all_series[1:]:
        common_idx = common_idx.intersection(s.dropna().index)

    forecasts_aligned = {k: v.loc[common_idx] for k, v in forecasts.items()}
    realised_aligned = realised.loc[common_idx]

    # Comparison table (exclude realized vol models from QLIKE scoring since
    # they are not forecasts in the parametric sense — include anyway for ref)
    table_df = comparison_table(forecasts_aligned, realised_aligned)
    comparison = {
        model: {metric: float(val) for metric, val in row.items()}
        for model, row in table_df.to_dict(orient="index").items()
    }

    return {
        "comparison_table": comparison,
        "forecasts": {k: _series_to_timeseries(v) for k, v in forecasts_aligned.items()},
        "realised": _series_to_timeseries(realised_aligned),
        "asset": req.asset,
        "freq": req.freq,
    }


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.post("/run")
async def run_vol_eval(req: VolEvalRequest, request: Request) -> dict[str, str]:
    """Submit a volatility model evaluation job."""
    store: JobStore = _get_store(request)
    job_id = store.create()
    asyncio.create_task(
        store.run(job_id, asyncio.to_thread(_run_vol_eval_sync, req))
    )
    return {"job_id": job_id}


@router.get("/{job_id}", response_model=EvalJobStatusResponse)
async def get_vol_eval_status(job_id: str, request: Request) -> EvalJobStatusResponse:
    """Poll vol eval job status."""
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
