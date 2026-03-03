from __future__ import annotations

from typing import Literal

from pydantic import BaseModel


class ModelConfig(BaseModel):
    model_id: str
    model_type: Literal["threshold", "vol_quantile", "hmm"]
    params: dict = {}


class RegimeDetectionRequest(BaseModel):
    assets: list[str] = ["BTC", "ETH"]
    freq: str = "1D"
    start_date: str | None = None
    end_date: str | None = None
    models: list[ModelConfig] = []
    prob_window: int = 20


class RegimeModelResult(BaseModel):
    model_id: str
    model_type: str
    unique_regimes: list[str]
    labels: list[tuple[str, str]]
    probabilities: dict[str, list[tuple[str, float]]]


class RegimeDetectionResult(BaseModel):
    assets: list[str]
    freq: str
    date_range: tuple[str, str]
    prices: dict[str, list[tuple[str, float]]]
    volumes: dict[str, list[tuple[str, float]]]
    models: list[RegimeModelResult]


class RegimeDetectionJobResponse(BaseModel):
    job_id: str
    status: str
    result: dict | None = None
    error: str | None = None
