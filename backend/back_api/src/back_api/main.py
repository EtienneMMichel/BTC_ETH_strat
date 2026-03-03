from __future__ import annotations

from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI
from fastapi.middleware.gzip import GZipMiddleware

from back_api.jobs import JobStore
from back_api.routers import backtest as backtest_router
from back_api.routers import co_mov as co_mov_router
from back_api.routers import evaluation as evaluation_router
from back_api.routers import price_forecast as price_forecast_router
from back_api.routers import regime_detection as regime_detection_router
from back_api.routers import vol_eval as vol_eval_router


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    app.state.job_store = JobStore()
    yield
    # Nothing to clean up for an in-memory store.


app = FastAPI(
    title="BTC/ETH Strategy API",
    description="Async job-queue API exposing backtest and evaluation from core.",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(GZipMiddleware, minimum_size=1024)

app.include_router(backtest_router.router, prefix="/backtest", tags=["backtest"])
app.include_router(evaluation_router.router, prefix="/evaluation", tags=["evaluation"])
app.include_router(vol_eval_router.router, prefix="/vol-eval", tags=["vol-eval"])
app.include_router(price_forecast_router.router, prefix="/price-forecast", tags=["price-forecast"])
app.include_router(co_mov_router.router, prefix="/co-mov", tags=["co-mov"])
app.include_router(regime_detection_router.router, prefix="/regime-detection", tags=["regime"])


@app.get("/health", tags=["meta"])
async def health() -> dict[str, str]:
    return {"status": "ok"}
