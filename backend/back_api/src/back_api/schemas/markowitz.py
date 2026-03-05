from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


# ── Covariance Eval ──────────────────────────────────────────────────────────

class CovEvalRequest(BaseModel):
    freq: str = "1D"
    models: list[Literal["rolling", "diagonal", "bekk"]] = ["rolling", "diagonal", "bekk"]
    window: int = Field(default=63, ge=5)
    min_history: int = Field(default=252, ge=10)


class CovModelSeries(BaseModel):
    var_BTC: list[tuple[str, float]]
    var_ETH: list[tuple[str, float]]
    cov_BTC_ETH: list[tuple[str, float]]


class CovEvalResult(BaseModel):
    models: dict[str, CovModelSeries]   # "rolling" | "diagonal" | "bekk"
    realised: CovModelSeries            # rolling(21) sample cov proxy
    comparison: dict[str, dict[str, float]]  # model → {rmse_var_BTC, rmse_var_ETH, corr_cov}


# ── Expected Returns Eval ────────────────────────────────────────────────────

class ERCovEvalRequest(BaseModel):
    freq: str = "1D"
    models: list[Literal["rolling_mean", "signal_tsmom", "signal_momentum"]] = [
        "rolling_mean", "signal_tsmom"
    ]
    er_window: int = Field(default=63, ge=5)
    min_history: int = Field(default=100, ge=10)


class ERModelMetrics(BaseModel):
    ic: float        # corr(predicted_mu_t, realized_return_{t+1})
    hit_rate: float  # sign accuracy


class ERAssetResult(BaseModel):
    mu: dict[str, list[tuple[str, float]]]  # model → timeseries
    metrics: dict[str, ERModelMetrics]


class ERCovEvalResult(BaseModel):
    per_asset: dict[str, ERAssetResult]   # "BTC", "ETH"


# ── Shared job response ───────────────────────────────────────────────────────

class MarkowitzJobResponse(BaseModel):
    job_id: str
    status: str
    result: CovEvalResult | ERCovEvalResult | None = None
    error: str | None = None
