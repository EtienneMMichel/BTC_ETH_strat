from __future__ import annotations

import asyncio
from typing import Any

import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException, Request

from back_api.config import get_settings
from back_api.jobs import JobStore
from back_api.schemas.evaluation import EvalJobStatusResponse
from back_api.schemas.price_forecast import PriceForecastRequest

router = APIRouter()

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

HORIZON_MAP: dict[str, str] = {
    "1W": "W",
    "3D": "3D",
    "1D": "1D",
    "8h": "8h",
    "4h": "4h",
    "1h": "1h",
}

# Approximate nanoseconds per resolution (for finer/coarser comparison)
_NS = 1_000_000_000
OFFSET_NS: dict[str, int] = {
    "1min":  60 * _NS,
    "5min":  5 * 60 * _NS,
    "15min": 15 * 60 * _NS,
    "1h":    3600 * _NS,
    "4h":    4 * 3600 * _NS,
    "8h":    8 * 3600 * _NS,
    "1D":    24 * 3600 * _NS,
    "3D":    3 * 24 * 3600 * _NS,
    "W":     7 * 24 * 3600 * _NS,
}

OUTPUT_TYPES: dict[str, str] = {
    "logistic":     "probability",
    "hp_filter":    "direction",
    "tsmom":        "amplitude",
    "momentum":     "amplitude",
    "ema_crossover": "amplitude",
    "kalman":       "amplitude",
}

RESOLUTION_FINE = {"1min", "5min", "15min", "1h"}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _get_store(request: Request) -> JobStore:
    return request.app.state.job_store


def _series_to_timeseries(s: pd.Series) -> list[list]:
    return [[ts.isoformat(), float(v)] for ts, v in s.dropna().items()]


def _get_model(name: str):
    """Lazy-import and instantiate a model by name. Returns None if unknown."""
    if name == "tsmom":
        from core.models.forecast.price.momentum import TSMOMModel
        return TSMOMModel()
    elif name == "momentum":
        from core.models.forecast.price.momentum import MomentumModel
        return MomentumModel()
    elif name == "ema_crossover":
        from core.models.forecast.price.trend import EMACrossover
        return EMACrossover()
    elif name == "hp_filter":
        from core.models.forecast.price.trend import HPFilter
        return HPFilter()
    elif name == "kalman":
        from core.models.forecast.price.trend import KalmanTrend
        return KalmanTrend()
    elif name == "logistic":
        from core.models.forecast.price.probability import LogisticModel
        return LogisticModel()
    return None


def _align(sig: pd.Series, model_res: str, horizon_alias: str) -> pd.Series:
    """Resample signal from model_res to horizon_alias."""
    if model_res == horizon_alias:
        return sig
    mo_ns = OFFSET_NS.get(model_res, 0)
    ho_ns = OFFSET_NS.get(horizon_alias, 0)
    if mo_ns == 0 or ho_ns == 0:
        return sig
    if mo_ns <= ho_ns:
        # finer → coarser: take last signal in each horizon bar
        return sig.resample(horizon_alias).last()
    else:
        # coarser → finer: forward-fill
        return sig.resample(horizon_alias).ffill()


def _common(sig: pd.Series, ret: pd.Series) -> tuple[pd.Series, pd.Series]:
    idx = sig.dropna().index.intersection(ret.dropna().index)
    return sig.loc[idx], ret.loc[idx]


# ── Direction metrics ────────────────────────────────────────────────────────

def _hit_rate(sig: pd.Series, ret: pd.Series) -> float:
    s, r = _common(sig, ret)
    if len(s) < 2:
        return float("nan")
    return float((np.sign(s.values) == np.sign(r.values)).mean())


def _confusion_counts(sig: pd.Series, ret: pd.Series) -> dict[str, int]:
    s, r = _common(sig, ret)
    if len(s) < 2:
        return {"tp": 0, "fp": 0, "tn": 0, "fn": 0}
    pred_pos = s.values > 0
    actual_pos = r.values > 0
    return {
        "tp": int((pred_pos & actual_pos).sum()),
        "fp": int((pred_pos & ~actual_pos).sum()),
        "tn": int((~pred_pos & ~actual_pos).sum()),
        "fn": int((~pred_pos & actual_pos).sum()),
    }


def _precision_recall_f1(cm: dict[str, int]) -> tuple[float, float, float]:
    tp, fp, fn = cm["tp"], cm["fp"], cm["fn"]
    prec = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
    rec = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
    if np.isnan(prec) or np.isnan(rec) or (prec + rec) == 0:
        f1 = float("nan")
    else:
        f1 = 2 * prec * rec / (prec + rec)
    return prec, rec, f1


def _roc_auc(sig: pd.Series, ret: pd.Series) -> tuple[list[list[float]], float]:
    _fallback: tuple[list[list[float]], float] = ([[0.0, 0.0], [1.0, 1.0]], 0.5)
    s, r = _common(sig, ret)
    if len(s) < 2:
        return _fallback
    actual = (r.values > 0).astype(int)
    n_pos = int(actual.sum())
    n_neg = len(actual) - n_pos
    if n_pos == 0 or n_neg == 0:
        return _fallback

    order = np.argsort(-s.values)
    actual_sorted = actual[order]

    points: list[list[float]] = [[0.0, 0.0]]
    tp, fp = 0, 0
    for a in actual_sorted:
        if a == 1:
            tp += 1
        else:
            fp += 1
        points.append([fp / n_neg, tp / n_pos])
    points.append([1.0, 1.0])

    if len(points) > 200:
        step = max(1, len(points) // 199)
        points = points[::step]
        if points[-1] != [1.0, 1.0]:
            points.append([1.0, 1.0])

    auc = 0.0
    for i in range(1, len(points)):
        dx = points[i][0] - points[i - 1][0]
        auc += dx * (points[i][1] + points[i - 1][1]) / 2

    return [[round(p[0], 4), round(p[1], 4)] for p in points], round(auc, 4)


# ── Amplitude metrics ────────────────────────────────────────────────────────

def _ic(sig: pd.Series, ret: pd.Series) -> float:
    s, r = _common(sig, ret)
    if len(s) < 2:
        return float("nan")
    sv, rv = s.values.astype(float), r.values.astype(float)
    if np.std(sv) < 1e-12 or np.std(rv) < 1e-12:
        return float("nan")
    return float(np.corrcoef(sv, rv)[0, 1])


def _rank_ic(sig: pd.Series, ret: pd.Series) -> float:
    from scipy.stats import spearmanr
    s, r = _common(sig, ret)
    if len(s) < 2:
        return float("nan")
    corr, _ = spearmanr(s.values, r.values)
    return float(corr)


def _calibration_bins(sig: pd.Series, ret: pd.Series, n: int) -> list[dict]:
    s, r = _common(sig, ret)
    if len(s) < n:
        return []
    try:
        abs_sig = s.abs()
        abs_ret = r.abs()
        bins = pd.qcut(abs_sig, q=n, duplicates="drop")
        result = []
        for b in bins.cat.categories:
            mask = bins == b
            result.append({
                "bin": float(b.mid),
                "mean_abs_signal": float(abs_sig[mask].mean()),
                "mean_abs_return": float(abs_ret[mask].mean()),
                "count": int(mask.sum()),
            })
        return result
    except Exception:
        return []


# ── Probability metrics ──────────────────────────────────────────────────────

def _reliability_diagram(sig: pd.Series, ret: pd.Series, n: int) -> list[dict]:
    """For [0,1] signals: bucket P into n bins, compute actual_rate per bin."""
    s, r = _common(sig, ret)
    if len(s) < n:
        return []
    try:
        bins = pd.cut(s, bins=n, include_lowest=True)
        y = (r > 0).astype(float)
        result = []
        for b in bins.cat.categories:
            mask = bins == b
            if mask.sum() == 0:
                continue
            result.append({
                "prob_bin": float(b.mid),
                "actual_rate": float(y[mask].mean()),
                "count": int(mask.sum()),
            })
        return result
    except Exception:
        return []


def _brier_score(sig: pd.Series, ret: pd.Series) -> float:
    s, r = _common(sig, ret)
    if len(s) == 0:
        return float("nan")
    y = (r.values > 0).astype(float)
    return float(np.mean((s.values - y) ** 2))


def _log_loss(sig: pd.Series, ret: pd.Series) -> float:
    s, r = _common(sig, ret)
    if len(s) == 0:
        return float("nan")
    y = (r.values > 0).astype(float)
    p = np.clip(s.values, 1e-9, 1 - 1e-9)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


# ── Additional metrics ───────────────────────────────────────────────────────

def _signal_histogram(sig: pd.Series, n_bins: int = 50) -> list[dict]:
    vals = sig.dropna().values
    if len(vals) == 0:
        return []
    counts, edges = np.histogram(vals, bins=n_bins)
    centers = (edges[:-1] + edges[1:]) / 2
    return [{"bin_center": float(c), "count": int(v)} for c, v in zip(centers, counts)]


def _signal_turnover(sig: pd.Series) -> float:
    s = sig.dropna()
    if len(s) < 2:
        return float("nan")
    signs = np.sign(s.values)
    changes = int((signs[1:] != signs[:-1]).sum())
    return float(changes / (len(signs) - 1))


def _signal_acf(sig: pd.Series, max_lags: int = 20) -> list[dict]:
    s = sig.dropna().values
    if len(s) < max_lags + 1:
        return []
    mean = float(np.mean(s))
    centered = s - mean
    var = float(np.sum(centered ** 2))
    if var < 1e-12:
        return []
    result = []
    for lag in range(1, max_lags + 1):
        cov = float(np.sum(centered[lag:] * centered[:-lag]))
        result.append({"lag": lag, "acf": round(cov / var, 6)})
    return result


# ---------------------------------------------------------------------------
# Main sync worker
# ---------------------------------------------------------------------------

def _run_price_forecast_sync(req: PriceForecastRequest) -> dict[str, Any]:
    from core.backtest.data.loader import load_assets, resample_ohlcv

    settings = get_settings()
    data_dir = req.data_dir if req.data_dir else str(settings.data_dir)

    # Step 1 — Load raw 1-min data (always)
    raw_1min_multi = load_assets(data_dir, req.assets, freq="1min")
    asset_ohlcv_1min: pd.DataFrame = raw_1min_multi[req.asset]

    # Step 3 — Forward returns at forecast_horizon
    horizon_alias = HORIZON_MAP.get(req.forecast_horizon, req.forecast_horizon)
    horizon_ohlcv = resample_ohlcv(asset_ohlcv_1min, horizon_alias)
    close_h = horizon_ohlcv["close"]
    next_ret = np.log(close_h).diff().shift(-1)  # forward return at horizon

    # Step 2 — Per-model resample + fit + predict
    signals: dict[str, pd.Series] = {}
    model_resolutions_used: dict[str, str] = {}
    warn_list: list[str] = []

    for model_name in req.models:
        res = req.model_resolutions.get(model_name, "1D")
        model_ohlcv = resample_ohlcv(asset_ohlcv_1min, res)
        close_at_res = model_ohlcv["close"]

        if model_name == "hp_filter" and res in RESOLUTION_FINE and len(close_at_res) > 500:
            warn_list.append(
                f"hp_filter skipped at {res} (n={len(close_at_res)} > 500, O(n²) cost)"
            )
            continue

        m = _get_model(model_name)
        if m is None:
            warn_list.append(f"Unknown model: {model_name}")
            continue

        m.fit(close_at_res)
        signals[model_name] = m.predict(close_at_res)
        model_resolutions_used[model_name] = res

    # Step 4 — Align signals + compute 3-section metrics
    metrics: dict[str, Any] = {}
    for model_name, sig in signals.items():
        res = model_resolutions_used[model_name]
        sig_aligned = _align(sig, res, horizon_alias)

        output_type = OUTPUT_TYPES.get(model_name, "amplitude")

        # — Direction section —
        cm = _confusion_counts(sig_aligned, next_ret)
        prec, rec, f1 = _precision_recall_f1(cm)
        roc_pts, auc = _roc_auc(sig_aligned, next_ret)
        direction: dict[str, Any] = {
            "hit_rate": _hit_rate(sig_aligned, next_ret),
            "precision": prec,
            "recall": rec,
            "f1": f1,
            "confusion": cm,
            "roc_curve": roc_pts,
            "auc": auc,
        }

        # — Amplitude section —
        ic = _ic(sig_aligned, next_ret)
        rank_ic = _rank_ic(sig_aligned, next_ret)
        s_a, r_a = _common(sig_aligned, next_ret)
        mae = float(np.abs(s_a.values - r_a.values).mean()) if len(s_a) > 0 else float("nan")
        mse = float(((s_a.values - r_a.values) ** 2).mean()) if len(s_a) > 0 else float("nan")
        amplitude: dict[str, Any] = {
            "ic": ic,
            "rank_ic": rank_ic,
            "mae": mae,
            "mse": mse,
            "calibration_bins": _calibration_bins(sig_aligned, next_ret, req.n_calibration_bins),
            "reliability_diagram": (
                _reliability_diagram(sig_aligned, next_ret, req.n_calibration_bins)
                if output_type == "probability" else None
            ),
            "brier_score": _brier_score(sig_aligned, next_ret) if output_type == "probability" else None,
            "log_loss": _log_loss(sig_aligned, next_ret) if output_type == "probability" else None,
        }

        # — Additional section —
        additional: dict[str, Any] = {
            "signal_histogram": _signal_histogram(sig, 50),
            "turnover": _signal_turnover(sig),
            "acf": _signal_acf(sig, 20),
        }

        metrics[model_name] = {
            "direction": direction,
            "amplitude": amplitude,
            "additional": additional,
            "output_type": output_type,
        }

    return {
        "signals": {k: _series_to_timeseries(v) for k, v in signals.items()},
        "metrics": metrics,
        "prices": _series_to_timeseries(close_h),
        "asset": req.asset,
        "forecast_horizon": req.forecast_horizon,
        "model_resolutions": model_resolutions_used,
        "warnings": warn_list,
    }


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.post("/run")
async def run_price_forecast(req: PriceForecastRequest, request: Request) -> dict[str, str]:
    """Submit a price forecast evaluation job."""
    store: JobStore = _get_store(request)
    job_id = store.create()
    asyncio.create_task(
        store.run(job_id, asyncio.to_thread(_run_price_forecast_sync, req))
    )
    return {"job_id": job_id}


@router.get("/{job_id}", response_model=EvalJobStatusResponse)
async def get_price_forecast_status(job_id: str, request: Request) -> EvalJobStatusResponse:
    """Poll price forecast job status."""
    store: JobStore = _get_store(request)
    rec = store.get(job_id)
    if rec is None:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found.")
    return EvalJobStatusResponse(
        job_id=job_id,
        status=rec.status,
        result=rec.result if rec.status == "done" else None,
        error=rec.error,
    )
