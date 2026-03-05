from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from back_api.schemas.backtest import MarkowitzParams


class StrategyConfig(BaseModel):
    label: str = ""
    strategy: Literal["momentum", "mean_reversion", "orchestrator", "markowitz"] = "orchestrator"
    markowitz: MarkowitzParams = Field(default_factory=MarkowitzParams)


class CompareRequest(BaseModel):
    data_dir: str = ""
    assets: dict[str, str] = Field(
        default={"BTC": "BTCUSDT", "ETH": "ETHUSDT"},
    )
    freq: str = Field(default="1D")
    min_train_bars: int = Field(default=252, ge=10)
    rebalance_every: int = Field(default=21, ge=1)
    fee_rate: float = Field(default=0.001, ge=0.0)
    slippage: float = Field(default=0.0005, ge=0.0)
    strategies: list[StrategyConfig] = Field(min_length=1, max_length=10)
    include_frontier: bool = True
    frontier_points: int = Field(default=20, ge=5, le=100)
    frontier_cov_model: Literal["rolling", "diagonal", "bekk"] = "diagonal"
    frontier_er_model: Literal["rolling_mean", "signal_tsmom"] = "signal_tsmom"
    frontier_long_only: bool = False


class StrategyResultEntry(BaseModel):
    label: str
    strategy: str
    equity_curve: list[tuple[str, float]]
    metrics: dict[str, float]
    trade_count: int
    signals: list[tuple[str, dict[str, float]]]
    ann_return: float
    ann_vol: float


class FrontierPoint(BaseModel):
    gamma: float
    ann_return: float
    ann_vol: float
    sharpe: float
    weights: dict[str, float]


class CompareResult(BaseModel):
    strategies: list[StrategyResultEntry]
    benchmarks: dict[str, list[tuple[str, float]]]
    frontier: list[FrontierPoint] | None = None
    frontier_special: dict[str, FrontierPoint] | None = None


class CompareJobResponse(BaseModel):
    job_id: str
    status: str
    result: CompareResult | None = None
    error: str | None = None
