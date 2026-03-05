from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class MarkowitzParams(BaseModel):
    objective: Literal["max_sharpe", "min_variance", "mean_variance", "max_diversification"] = "max_sharpe"
    cov_model: Literal["rolling", "diagonal", "bekk"] = "diagonal"
    er_model: Literal["rolling_mean", "signal_tsmom"] = "signal_tsmom"
    gamma: float = Field(default=1.0, ge=0.01)
    long_only: bool = False
    max_weight: float = Field(default=1.0, ge=0.0, le=2.0)
    min_weight: float = Field(default=-1.0, ge=-2.0, le=0.0)
    risk_free_rate: float = 0.0
    target_vol: float | None = None
    min_history: int = Field(default=252, ge=10)


class BacktestRequest(BaseModel):
    data_dir: str = Field(
        description=(
            "Root directory of raw parquet data (overrides server default). "
            "If omitted, the server-configured data_dir is used."
        ),
        default="",
    )
    assets: dict[str, str] = Field(
        default={"BTC": "BTCUSDT", "ETH": "ETHUSDT"},
        description='Mapping from pipeline name to Binance symbol, e.g. {"BTC": "BTCUSDT"}',
    )
    freq: str = Field(default="1D", description='Pandas offset string: "1D", "4h", "1h", …')
    strategy: Literal["momentum", "mean_reversion", "orchestrator", "markowitz"] = "orchestrator"
    min_train_bars: int = Field(default=252, ge=10)
    rebalance_every: int = Field(default=21, ge=1)
    fee_rate: float = Field(default=0.001, ge=0.0)
    slippage: float = Field(default=0.0005, ge=0.0)
    markowitz: MarkowitzParams = Field(default_factory=MarkowitzParams)


class BacktestResultSchema(BaseModel):
    equity_curve: list[tuple[str, float]]
    """List of (ISO timestamp, NAV) pairs."""

    benchmarks: dict[str, list[tuple[str, float]]] = Field(default_factory=dict)
    """Per-asset buy-and-hold cumulative returns, e.g. {"BTC": [...], "ETH": [...]}."""

    metrics: dict[str, float]
    """sharpe, max_drawdown, calmar, var, es, win_rate."""

    trade_count: int
    signals: list[tuple[str, dict[str, float]]]
    """List of (ISO timestamp, {asset: weight}) pairs at each rebalance."""


class JobStatusResponse(BaseModel):
    job_id: str
    status: str
    result: BacktestResultSchema | None = None
    error: str | None = None
