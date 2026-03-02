from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class VolEvalRequest(BaseModel):
    asset: Literal["BTC", "ETH"] = "BTC"
    freq: str = "1D"
    min_train_bars: int = Field(252, ge=20)
    test_size: int = Field(21, ge=1)
    models: list[str] = ["garch", "gjr_garch", "egarch", "ewma", "rogers_satchell", "yang_zhang"]
    data_dir: str = ""
    assets: dict[str, str] = {"BTC": "BTCUSDT", "ETH": "ETHUSDT"}
