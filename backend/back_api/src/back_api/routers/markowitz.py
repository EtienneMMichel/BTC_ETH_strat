from __future__ import annotations

import asyncio
import math
from typing import Any

import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException, Request

from back_api.config import get_settings
from back_api.jobs import JobStore
from back_api.schemas.markowitz import (
    CovEvalRequest,
    CovEvalResult,
    CovModelSeries,
    ERCovEvalRequest,
    ERCovEvalResult,
    ERAssetResult,
    ERModelMetrics,
    MarkowitzJobResponse,
)

router = APIRouter()


def _get_store(request: Request) -> JobStore:
    return request.app.state.job_store


# ---------------------------------------------------------------------------
# Helper: load close prices
# ---------------------------------------------------------------------------

def _load_prices(freq: str) -> pd.DataFrame:
    from core.backtest.data.loader import load_assets

    settings = get_settings()
    assets = {"BTC": "BTCUSDT", "ETH": "ETHUSDT"}
    ohlcv = load_assets(str(settings.data_dir), assets, freq)
    return pd.DataFrame({name: ohlcv[name]["close"] for name in assets})


def _series_to_list(s: pd.Series) -> list[tuple[str, float]]:
    """Convert a Series to [(iso_str, float)] dropping NaN/inf."""
    result = []
    for ts, val in s.items():
        fval = float(val)
        if math.isfinite(fval):
            result.append((ts.isoformat(), fval))
    return result


# ---------------------------------------------------------------------------
# Covariance evaluation worker
# ---------------------------------------------------------------------------

def _run_cov_eval_sync(req: CovEvalRequest) -> CovEvalResult:
    from core.models.covariance.rolling import RollingCovModel
    from core.models.covariance.diagonal import DiagonalCovModel
    from core.models.covariance.bekk_cov import BEKKCovModel

    prices = _load_prices(req.freq)
    log_returns = np.log(prices).diff().dropna()

    model_map = {
        "rolling": lambda: RollingCovModel(window=req.window),
        "diagonal": DiagonalCovModel,
        "bekk": BEKKCovModel,
    }

    # Realised proxy: rolling(21)
    realised_model = RollingCovModel(window=21)
    realised_model.fit(log_returns)
    realised_pred = realised_model.predict(log_returns).iloc[req.min_history:]

    realised_series = CovModelSeries(
        var_BTC=_series_to_list(realised_pred["var_BTC"]),
        var_ETH=_series_to_list(realised_pred["var_ETH"]),
        cov_BTC_ETH=_series_to_list(realised_pred["cov_BTC_ETH"]),
    )

    models_out: dict[str, CovModelSeries] = {}
    comparison: dict[str, dict[str, float]] = {}

    for name in req.models:
        m = model_map[name]()
        m.fit(log_returns)
        pred = m.predict(log_returns).iloc[req.min_history:]

        models_out[name] = CovModelSeries(
            var_BTC=_series_to_list(pred["var_BTC"]),
            var_ETH=_series_to_list(pred["var_ETH"]),
            cov_BTC_ETH=_series_to_list(pred["cov_BTC_ETH"]),
        )

        # Align on common index for comparison
        common_idx = pred.index.intersection(realised_pred.index)
        if len(common_idx) > 0:
            p_aligned = pred.loc[common_idx]
            r_aligned = realised_pred.loc[common_idx]

            rmse_var_btc = float(np.sqrt(np.mean((p_aligned["var_BTC"] - r_aligned["var_BTC"]) ** 2)))
            rmse_var_eth = float(np.sqrt(np.mean((p_aligned["var_ETH"] - r_aligned["var_ETH"]) ** 2)))

            # Pearson corr of cov_BTC_ETH series
            pc = p_aligned["cov_BTC_ETH"]
            rc = r_aligned["cov_BTC_ETH"]
            if pc.std() > 0 and rc.std() > 0:
                corr_cov = float(np.corrcoef(pc.values, rc.values)[0, 1])
            else:
                corr_cov = float("nan")
        else:
            rmse_var_btc = float("nan")
            rmse_var_eth = float("nan")
            corr_cov = float("nan")

        comparison[name] = {
            "rmse_var_BTC": rmse_var_btc,
            "rmse_var_ETH": rmse_var_eth,
            "corr_cov": corr_cov,
        }

    return CovEvalResult(
        models=models_out,
        realised=realised_series,
        comparison=comparison,
    )


# ---------------------------------------------------------------------------
# Expected Returns evaluation worker
# ---------------------------------------------------------------------------

def _run_er_eval_sync(req: ERCovEvalRequest) -> ERCovEvalResult:
    from core.models.expected_returns.rolling_mean import RollingMeanReturns
    from core.models.expected_returns.signal import SignalExpectedReturns
    from core.models.forecast.price.momentum import TSMOMModel, MomentumModel

    prices = _load_prices(req.freq)
    log_returns = np.log(prices).diff().dropna()

    def _make_model(name: str) -> Any:
        if name == "rolling_mean":
            return RollingMeanReturns(window=req.er_window)
        elif name == "signal_tsmom":
            return SignalExpectedReturns(model_cls=TSMOMModel)
        elif name == "signal_momentum":
            return SignalExpectedReturns(model_cls=MomentumModel)
        else:
            raise ValueError(f"Unknown ER model: {name}")

    assets = ["BTC", "ETH"]
    per_asset: dict[str, ERAssetResult] = {}

    # Collect mu series for all models first
    mu_by_model: dict[str, pd.DataFrame] = {}
    for name in req.models:
        m = _make_model(name)
        m.fit(prices)
        mu = m.predict(prices).iloc[req.min_history:]
        mu_by_model[name] = mu

    for asset in assets:
        mu_series: dict[str, list[tuple[str, float]]] = {}
        metrics: dict[str, ERModelMetrics] = {}

        # Next-day realized returns aligned for IC computation
        next_ret = log_returns[asset].shift(-1)

        for name, mu_df in mu_by_model.items():
            mu_asset = mu_df[asset]
            mu_series[name] = _series_to_list(mu_asset)

            # Align mu with next-day returns
            common_idx = mu_asset.index.intersection(next_ret.dropna().index)
            if len(common_idx) > 1:
                mu_vals = mu_asset.loc[common_idx].values
                nr_vals = next_ret.loc[common_idx].values

                # IC = Pearson corr
                if mu_vals.std() > 0 and nr_vals.std() > 0:
                    ic = float(np.corrcoef(mu_vals, nr_vals)[0, 1])
                else:
                    ic = 0.0

                # Hit rate = sign accuracy
                hit_rate = float(np.mean(np.sign(mu_vals) == np.sign(nr_vals)))
            else:
                ic = 0.0
                hit_rate = 0.0

            metrics[name] = ERModelMetrics(ic=ic, hit_rate=hit_rate)

        per_asset[asset] = ERAssetResult(mu=mu_series, metrics=metrics)

    return ERCovEvalResult(per_asset=per_asset)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.post("/covariance-eval")
async def run_cov_eval(req: CovEvalRequest, request: Request) -> dict[str, str]:
    """Submit a covariance model evaluation job."""
    store: JobStore = _get_store(request)
    job_id = store.create()
    asyncio.create_task(
        store.run(job_id, asyncio.to_thread(_run_cov_eval_sync, req))
    )
    return {"job_id": job_id}


@router.post("/er-eval")
async def run_er_eval(req: ERCovEvalRequest, request: Request) -> dict[str, str]:
    """Submit an expected-returns evaluation job."""
    store: JobStore = _get_store(request)
    job_id = store.create()
    asyncio.create_task(
        store.run(job_id, asyncio.to_thread(_run_er_eval_sync, req))
    )
    return {"job_id": job_id}


@router.get("/{job_id}", response_model=MarkowitzJobResponse)
async def get_markowitz_job(job_id: str, request: Request) -> MarkowitzJobResponse:
    """Poll a markowitz evaluation job."""
    store: JobStore = _get_store(request)
    rec = store.get(job_id)
    if rec is None:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found.")
    return MarkowitzJobResponse(
        job_id=job_id,
        status=rec.status,
        result=rec.result if rec.status == "done" else None,
        error=rec.error,
    )
