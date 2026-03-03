"""
Regime Detection router.

POST /regime-detection/run  → {job_id}
GET  /regime-detection/{job_id} → RegimeDetectionJobResponse
"""

from __future__ import annotations

import asyncio
from typing import Any

import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException, Request

from back_api.config import get_settings
from back_api.jobs import JobStore
from back_api.schemas.regime_detection import (
    RegimeDetectionJobResponse,
    RegimeDetectionRequest,
)

router = APIRouter()

# Asset symbol mapping (matches price_forecast defaults)
_DEFAULT_SYMBOLS = {
    "BTC": "BTCUSDT",
    "ETH": "ETHUSDT",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _get_store(request: Request) -> JobStore:
    return request.app.state.job_store


def _series_to_ts(s: pd.Series) -> list[tuple[str, float]]:
    return [(ts.isoformat(), float(v)) for ts, v in s.dropna().items()]


def _labels_to_ts(labels: pd.Series) -> list[tuple[str, str]]:
    return [(ts.isoformat(), str(v)) for ts, v in labels.dropna().items()]


def _proba_to_ts(proba_col: pd.Series) -> list[tuple[str, float]]:
    return [(ts.isoformat(), float(v)) for ts, v in proba_col.items()]


def _rolling_freq_proba(
    labels: pd.Series,
    unique_regimes: list[str],
    window: int,
) -> dict[str, list[tuple[str, float]]]:
    """
    Convert a categorical label series to pseudo-probabilities via rolling
    frequency: pd.get_dummies(labels).rolling(window).mean()
    """
    dummies = pd.get_dummies(labels.astype(str))
    # Ensure all regimes present as columns
    for r in unique_regimes:
        if r not in dummies.columns:
            dummies[r] = 0.0
    rolled = dummies.rolling(window, min_periods=1).mean()
    result: dict[str, list[tuple[str, float]]] = {}
    for regime in unique_regimes:
        col = rolled[regime] if regime in rolled.columns else pd.Series(0.0, index=labels.index)
        result[regime] = _proba_to_ts(col)
    return result


# ---------------------------------------------------------------------------
# Per-model processing
# ---------------------------------------------------------------------------


def _process_threshold(
    close_btc: pd.Series,
    close_eth: pd.Series,
    params: dict,
    prob_window: int,
) -> tuple[pd.Series, dict[str, list[tuple[str, float]]]]:
    """Run RegimeClassifier on BTC+ETH prices → hard labels + rolling proba."""
    from core.strats.orchestrator.regime import RegimeClassifier

    prices = pd.DataFrame({"BTC": close_btc, "ETH": close_eth}).dropna()
    classifier = RegimeClassifier(
        vol_window=params.get("vol_window", 63),
        vol_threshold_pct=params.get("vol_threshold_pct", 0.6),
        adf_pvalue=params.get("adf_pvalue", 0.05),
        drawdown_threshold=params.get("drawdown_threshold", 0.15),
        min_holding_bars=params.get("min_holding_bars", 5),
        beta_window=params.get("beta_window", 252),
    )
    labels = classifier.classify(prices)
    unique_regimes = sorted(labels.unique().tolist())
    proba = _rolling_freq_proba(labels, unique_regimes, prob_window)
    return labels, proba, unique_regimes


def _process_vol_quantile(
    close: pd.Series,
    params: dict,
    prob_window: int,
) -> tuple[pd.Series, dict[str, list[tuple[str, float]]]]:
    """Rolling-vol quantile binning → hard labels + rolling proba."""
    n_regimes = params.get("n_regimes", 2)
    vol_window = params.get("vol_window", 21)

    log_ret = np.log(close).diff()
    vol = log_ret.rolling(vol_window).std()
    quantile_edges = np.linspace(0, 1, n_regimes + 1)
    bin_edges = vol.quantile(quantile_edges).values
    label_names = [f"regime_{i}" for i in range(n_regimes)]

    try:
        regime = pd.cut(vol, bins=bin_edges, labels=label_names, include_lowest=True)
        labels = regime.astype(object).where(regime.notna(), other=None)
        labels = labels.dropna().astype(str)
    except Exception:
        labels = pd.Series("regime_0", index=close.index, dtype=str)

    unique_regimes = sorted(set(label_names))
    proba = _rolling_freq_proba(labels, unique_regimes, prob_window)
    return labels, proba, unique_regimes


def _process_hmm(
    close_btc: pd.Series,
    close_eth: pd.Series,
    params: dict,
    _prob_window: int,
) -> tuple[pd.Series, dict[str, list[tuple[str, float]]]]:
    """Fit HMMRegimeClassifier → Viterbi labels + forward-backward posteriors."""
    from core.strats.orchestrator.hmm import HMMRegimeClassifier

    prices = pd.DataFrame({"BTC": close_btc, "ETH": close_eth}).dropna()
    clf = HMMRegimeClassifier(
        n_components=params.get("n_components", 2),
        covariance_type=params.get("covariance_type", "full"),
        n_iter=params.get("n_iter", 100),
        vol_window=params.get("vol_window", 5),
        random_state=params.get("random_state", 42),
    )
    clf.fit(prices)
    labels = clf.predict(prices)
    proba_df = clf.predict_proba(prices)

    unique_regimes = sorted(proba_df.columns.tolist())
    proba: dict[str, list[tuple[str, float]]] = {}
    for regime in unique_regimes:
        proba[regime] = _proba_to_ts(proba_df[regime])

    return labels, proba, unique_regimes


# ---------------------------------------------------------------------------
# Main sync worker
# ---------------------------------------------------------------------------


def _run_regime_detection_sync(req: RegimeDetectionRequest) -> dict[str, Any]:
    from core.backtest.data.loader import load_assets, resample_ohlcv

    settings = get_settings()
    data_dir = str(settings.data_dir)

    # Always load BTC + ETH (needed for threshold and HMM)
    symbols = {k: v for k, v in _DEFAULT_SYMBOLS.items() if k in ("BTC", "ETH")}
    raw_1min = load_assets(data_dir, symbols, freq="1min")

    # Resample to requested frequency
    freq = req.freq
    ohlcv: dict[str, pd.DataFrame] = {}
    for asset in ("BTC", "ETH"):
        ohlcv[asset] = resample_ohlcv(raw_1min[asset], freq)

    # Apply date range filter
    def _slice(df: pd.DataFrame) -> pd.DataFrame:
        if req.start_date:
            df = df[df.index >= req.start_date]
        if req.end_date:
            df = df[df.index <= req.end_date]
        return df

    for asset in ohlcv:
        ohlcv[asset] = _slice(ohlcv[asset])

    close_btc = ohlcv["BTC"]["close"]
    close_eth = ohlcv["ETH"]["close"]
    volume_btc = ohlcv["BTC"]["volume"]
    volume_eth = ohlcv["ETH"]["volume"]

    # Determine common date range
    common_idx = close_btc.index.intersection(close_eth.index)
    close_btc = close_btc.loc[common_idx]
    close_eth = close_eth.loc[common_idx]
    volume_btc = volume_btc.reindex(common_idx)
    volume_eth = volume_eth.reindex(common_idx)

    date_range = (
        common_idx[0].isoformat() if len(common_idx) > 0 else "",
        common_idx[-1].isoformat() if len(common_idx) > 0 else "",
    )

    # Build prices + volumes output
    prices_out: dict[str, list[tuple[str, float]]] = {
        "BTC": _series_to_ts(close_btc),
        "ETH": _series_to_ts(close_eth),
    }
    volumes_out: dict[str, list[tuple[str, float]]] = {
        "BTC": _series_to_ts(volume_btc),
        "ETH": _series_to_ts(volume_eth),
    }

    # Process each model config
    model_results = []
    for mc in req.models:
        try:
            if mc.model_type == "threshold":
                labels, proba, unique_regimes = _process_threshold(
                    close_btc, close_eth, mc.params, req.prob_window
                )
            elif mc.model_type == "vol_quantile":
                # Use BTC close as primary asset for vol_quantile
                labels, proba, unique_regimes = _process_vol_quantile(
                    close_btc, mc.params, req.prob_window
                )
            elif mc.model_type == "hmm":
                labels, proba, unique_regimes = _process_hmm(
                    close_btc, close_eth, mc.params, req.prob_window
                )
            else:
                continue

            model_results.append(
                {
                    "model_id": mc.model_id,
                    "model_type": mc.model_type,
                    "unique_regimes": unique_regimes,
                    "labels": _labels_to_ts(labels),
                    "probabilities": proba,
                }
            )
        except Exception as e:
            # Include model with empty data and error note in model_id
            model_results.append(
                {
                    "model_id": f"{mc.model_id} [ERROR: {e}]",
                    "model_type": mc.model_type,
                    "unique_regimes": [],
                    "labels": [],
                    "probabilities": {},
                }
            )

    return {
        "assets": req.assets,
        "freq": freq,
        "date_range": date_range,
        "prices": prices_out,
        "volumes": volumes_out,
        "models": model_results,
    }


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.post("/run")
async def run_regime_detection(
    req: RegimeDetectionRequest, request: Request
) -> dict[str, str]:
    """Submit a regime detection job."""
    store: JobStore = _get_store(request)
    job_id = store.create()
    asyncio.create_task(
        store.run(job_id, asyncio.to_thread(_run_regime_detection_sync, req))
    )
    return {"job_id": job_id}


@router.get("/{job_id}", response_model=RegimeDetectionJobResponse)
async def get_regime_detection_status(
    job_id: str, request: Request
) -> RegimeDetectionJobResponse:
    """Poll regime detection job status."""
    store: JobStore = _get_store(request)
    rec = store.get(job_id)
    if rec is None:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found.")
    return RegimeDetectionJobResponse(
        job_id=job_id,
        status=rec.status,
        result=rec.result if rec.status == "done" else None,
        error=rec.error,
    )
