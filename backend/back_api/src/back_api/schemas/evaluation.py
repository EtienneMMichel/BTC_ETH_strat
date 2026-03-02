from __future__ import annotations

from pydantic import BaseModel, Field


class VolComparisonRequest(BaseModel):
    forecasts: dict[str, list[float]] = Field(
        description="Model name → vol forecast series (same length as realised)."
    )
    realised: list[float] = Field(description="Realised volatility series.")
    index: list[str] = Field(description="ISO timestamp strings (same length as realised).")
    metrics: list[str] = Field(
        default=["qlike", "mse", "mae"],
        description='Any combination of "qlike", "mse", "mae".',
    )


class VarTestRequest(BaseModel):
    returns: list[float]
    var_series: list[float] = Field(description="VaR forecasts (negative numbers, same length as returns).")
    alpha: float = Field(default=0.05, gt=0.0, lt=1.0)
    index: list[str]


class DMTestRequest(BaseModel):
    forecast_a: list[float]
    forecast_b: list[float]
    realised: list[float]
    index: list[str]
    loss: str = Field(default="mse", description='"mse" or "mae".')
    h: int = Field(default=1, ge=1, description="Forecast horizon for Newey-West HAC bandwidth.")


class EvalJobStatusResponse(BaseModel):
    job_id: str
    status: str
    result: dict | None = None
    error: str | None = None
