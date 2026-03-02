from __future__ import annotations

from pydantic import BaseModel, Field


class SaveRunRequest(BaseModel):
    run_name: str = Field(min_length=1, max_length=128)


class SavedRunMeta(BaseModel):
    run_id: str
    run_name: str
    timestamp: str
    assets: list[str]
    forecast_horizon: str
    models: list[str]
    summary_metrics: dict
