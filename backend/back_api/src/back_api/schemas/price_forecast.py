from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class PriceForecastRequest(BaseModel):
    asset: Literal["BTC", "ETH"] = "BTC"
    forecast_horizon: str = "1D"  # "1h"|"4h"|"8h"|"1D"|"3D"|"1W"
    min_train_bars: int = Field(252, ge=20)
    models: list[str] = ["tsmom", "momentum", "ema_crossover", "hp_filter", "kalman"]
    model_resolutions: dict[str, str] = {}  # model_name → resolution; missing = "1D"
    n_calibration_bins: int = Field(10, ge=2, le=20)
    data_dir: str = ""
    assets: dict[str, str] = {"BTC": "BTCUSDT", "ETH": "ETHUSDT"}
