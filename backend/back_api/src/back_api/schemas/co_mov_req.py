from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class CoMovRequest(BaseModel):
    freq: str = "1D"
    rolling_window: int = Field(126, ge=30, le=252)
    copula_type: Literal["gaussian", "student_t", "clayton"] = "clayton"
    data_dir: str = ""
    assets: dict[str, str] = {"BTC": "BTCUSDT", "ETH": "ETHUSDT"}
