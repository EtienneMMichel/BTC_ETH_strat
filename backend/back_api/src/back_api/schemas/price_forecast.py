from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field, model_validator


class PriceForecastRequest(BaseModel):
    # Legacy single-asset field (backward-compat)
    asset: Optional[Literal["BTC", "ETH"]] = None
    # Multi-asset list (new)
    assets: list[Literal["BTC", "ETH"]] = Field(default_factory=lambda: ["BTC"])
    cross_asset: bool = False

    forecast_horizon: str = "1D"  # "1h"|"4h"|"8h"|"1D"|"3D"|"1W"
    models: list[str] = Field(
        default_factory=lambda: ["tsmom", "momentum", "ema_crossover", "hp_filter", "kalman"]
    )
    model_resolutions: dict[str, str] = Field(default_factory=dict)
    n_calibration_bins: int = Field(10, ge=2, le=20)
    data_dir: str = ""
    # Symbol map: asset name → exchange ticker
    asset_symbols: dict[str, str] = Field(
        default_factory=lambda: {"BTC": "BTCUSDT", "ETH": "ETHUSDT"}
    )

    # Feature 1 — Train/Test date split ("YYYY-MM-DD")
    train_start: Optional[str] = None
    train_end: Optional[str] = None
    test_start: Optional[str] = None
    test_end: Optional[str] = None

    # Feature 4 — Regime conditioning
    regime_conditioning: bool = False
    regime_classifier_type: Literal["threshold", "vol_quantile", "manual"] = "vol_quantile"
    regime_classifier_params: dict = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate(self) -> "PriceForecastRequest":
        # backward-compat: if legacy `asset` provided and `assets` is default ["BTC"], override
        if self.asset is not None and self.assets == ["BTC"]:
            self.assets = [self.asset]

        # date fields: either all 4 or none
        date_fields = [self.train_start, self.train_end, self.test_start, self.test_end]
        n_set = sum(v is not None for v in date_fields)
        if 0 < n_set < 4:
            raise ValueError(
                "Provide all four date fields (train_start, train_end, test_start, test_end) or none."
            )
        if n_set == 4:
            if self.train_start >= self.train_end:  # type: ignore[operator]
                raise ValueError("train_start must be before train_end")
            if self.test_start >= self.test_end:  # type: ignore[operator]
                raise ValueError("test_start must be before test_end")

        return self
