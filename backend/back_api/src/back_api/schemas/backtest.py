from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


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
    strategy: Literal["momentum", "mean_reversion", "orchestrator"] = "orchestrator"
    min_train_bars: int = Field(default=252, ge=10)
    rebalance_every: int = Field(default=21, ge=1)
    fee_rate: float = Field(default=0.001, ge=0.0)
    slippage: float = Field(default=0.0005, ge=0.0)


class BacktestResultSchema(BaseModel):
    equity_curve: list[tuple[str, float]]
    """List of (ISO timestamp, NAV) pairs."""

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
