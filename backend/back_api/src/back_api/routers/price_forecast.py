from __future__ import annotations

import asyncio
import gzip
import json
import pickle
import uuid
from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException, Request

from back_api.config import get_settings
from back_api.jobs import JobStore
from back_api.schemas.evaluation import EvalJobStatusResponse
from back_api.schemas.price_forecast import PriceForecastRequest
from back_api.schemas.saved_run import SaveRunRequest, SavedRunMeta

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
    "logistic":       "probability",
    "hp_filter":      "direction",
    "tsmom":          "amplitude",
    "momentum":       "amplitude",
    "ema_crossover":  "amplitude",
    "kalman":         "amplitude",
    "cross_tsmom":    "amplitude",
    "cross_momentum": "amplitude",
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
    """Lazy-import and instantiate a model by name. Strips cross_ prefix."""
    base = name.removeprefix("cross_")
    if base == "tsmom":
        from core.models.forecast.price.momentum import TSMOMModel
        return TSMOMModel()
    elif base == "momentum":
        from core.models.forecast.price.momentum import MomentumModel
        return MomentumModel()
    elif base == "ema_crossover":
        from core.models.forecast.price.trend import EMACrossover
        return EMACrossover()
    elif base == "hp_filter":
        from core.models.forecast.price.trend import HPFilter
        return HPFilter()
    elif base == "kalman":
        from core.models.forecast.price.trend import KalmanTrend
        return KalmanTrend()
    elif base == "logistic":
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
        return sig.resample(horizon_alias).last()
    else:
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


def _reliability_diagram(sig: pd.Series, ret: pd.Series, n: int) -> list[dict]:
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
# Core metric computation helper (shared by main loop + regime loop)
# ---------------------------------------------------------------------------

def _compute_metrics_for_slice(
    sig_aligned: pd.Series,
    next_ret: pd.Series,
    n_bins: int,
    output_type: str,
    sig_raw: pd.Series | None = None,
) -> dict[str, Any]:
    """Compute all 3-section metrics for a (signal, return) pair."""
    # Direction
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

    # Amplitude
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
        "calibration_bins": _calibration_bins(sig_aligned, next_ret, n_bins),
        "reliability_diagram": (
            _reliability_diagram(sig_aligned, next_ret, n_bins)
            if output_type == "probability" else None
        ),
        "brier_score": _brier_score(sig_aligned, next_ret) if output_type == "probability" else None,
        "log_loss": _log_loss(sig_aligned, next_ret) if output_type == "probability" else None,
    }

    # Additional (use raw signal for histogram/ACF if available)
    sig_for_additional = sig_raw if sig_raw is not None else sig_aligned
    additional: dict[str, Any] = {
        "signal_histogram": _signal_histogram(sig_for_additional, 50),
        "turnover": _signal_turnover(sig_for_additional),
        "acf": _signal_acf(sig_for_additional, 20),
    }

    return {
        "direction": direction,
        "amplitude": amplitude,
        "additional": additional,
        "output_type": output_type,
    }


# ---------------------------------------------------------------------------
# Feature 1 — Date split helper
# ---------------------------------------------------------------------------

def _apply_date_split(
    close_h: pd.Series,
    req: PriceForecastRequest,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Return (close_train, close_test, next_ret_test). If no dates, full series."""
    if req.train_start is None:
        next_ret = np.log(close_h).diff().shift(-1)
        return close_h, close_h, next_ret
    close_train = close_h[req.train_start : req.train_end]
    close_test = close_h[req.test_start : req.test_end]
    next_ret_test = np.log(close_test).diff().shift(-1)
    return close_train, close_test, next_ret_test


# ---------------------------------------------------------------------------
# Feature 4 — Regime classifier helpers
# ---------------------------------------------------------------------------

def _classify_vol_quantile(
    close: pd.Series,
    n_regimes: int = 2,
    vol_window: int = 21,
) -> pd.Series:
    log_ret = np.log(close).diff()
    vol = log_ret.rolling(vol_window).std()
    quantiles = np.linspace(0, 1, n_regimes + 1)
    labels = [f"regime_{i}" for i in range(n_regimes)]
    try:
        bins_q = vol.quantile(quantiles).values
        regime = pd.cut(vol, bins=bins_q, labels=labels, include_lowest=True)
        return regime.astype(object).where(regime.notna(), other=None)
    except Exception:
        return pd.Series([None] * len(close), index=close.index, dtype=object)


def _classify_threshold(
    btc_close: pd.Series,
    eth_close: pd.Series,
    params: dict,
) -> pd.Series:
    from core.strats.orchestrator.regime import RegimeClassifier

    classifier = RegimeClassifier(
        vol_window=params.get("vol_window", 63),
        vol_threshold_pct=params.get("vol_threshold_pct", 0.6),
        drawdown_threshold=params.get("drawdown_threshold", -0.2),
        adf_pvalue=params.get("adf_pvalue", 0.05),
        min_holding_bars=params.get("min_holding_bars", 5),
    )
    btc_ret = np.log(btc_close).diff().dropna()
    eth_ret = np.log(eth_close).diff().dropna()
    common_idx = btc_ret.index.intersection(eth_ret.index)
    returns = pd.DataFrame({"BTC": btc_ret.loc[common_idx], "ETH": eth_ret.loc[common_idx]})

    labels = []
    for i in range(len(returns)):
        regime = classifier.classify(returns.iloc[: i + 1])
        labels.append(regime.value if hasattr(regime, "value") else str(regime))
    result = pd.Series(labels, index=returns.index)
    return result.reindex(btc_close.index)


def _classify_manual(index: pd.DatetimeIndex, date_ranges: list[dict]) -> pd.Series:
    result = pd.Series([None] * len(index), index=index, dtype=object)
    for dr in date_ranges:
        label = dr.get("label", "regime")
        start = dr.get("start")
        end = dr.get("end")
        if start and end:
            mask = (index >= start) & (index <= end)
            result.loc[mask] = label
    return result


# ---------------------------------------------------------------------------
# Feature 3 — Save helper
# ---------------------------------------------------------------------------

def _nan_to_none(obj: Any) -> Any:
    """Recursively convert float NaN/Inf to None for JSON safety."""
    if isinstance(obj, float):
        return None if (np.isnan(obj) or np.isinf(obj)) else obj
    if isinstance(obj, dict):
        return {k: _nan_to_none(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_nan_to_none(v) for v in obj]
    return obj


def _save_run_sync(
    run_id: str,
    run_name: str,
    result: dict[str, Any],
    settings: Any,
) -> SavedRunMeta:
    run_dir = settings.saved_runs_dir / run_id
    models_dir = run_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    # Extract fitted models (non-JSON-serializable) before saving
    fitted_models: dict[str, dict] = result.pop("_fitted_models", {})
    for asset_name, model_map in fitted_models.items():
        for model_name, model_obj in model_map.items():
            pkl_path = models_dir / f"{asset_name}_{model_name}.pkl"
            with open(pkl_path, "wb") as f:
                pickle.dump(model_obj, f)

    # Gzip-compress full result
    result_path = run_dir / "result.json.gz"
    safe_result = _nan_to_none(result)
    with gzip.open(result_path, "wt", encoding="utf-8") as f:
        json.dump(safe_result, f)

    # Build summary metrics
    summary_metrics: dict = {}
    per_asset = result.get("per_asset", {})
    for asset_name, asset_data in per_asset.items():
        summary_metrics[asset_name] = {}
        for model_name, m in asset_data.get("metrics", {}).items():
            summary_metrics[asset_name][model_name] = {
                "hit_rate": m.get("direction", {}).get("hit_rate"),
                "ic": m.get("amplitude", {}).get("ic"),
                "auc": m.get("direction", {}).get("auc"),
            }

    all_models = sorted({m for am in summary_metrics.values() for m in am})
    meta = SavedRunMeta(
        run_id=run_id,
        run_name=run_name,
        timestamp=datetime.now(timezone.utc).isoformat(),
        assets=result.get("assets", []),
        forecast_horizon=result.get("forecast_horizon", ""),
        models=all_models,
        summary_metrics=_nan_to_none(summary_metrics),
    )
    meta_path = run_dir / "metadata.json"
    with open(meta_path, "w") as f:
        json.dump(meta.model_dump(), f, indent=2)

    return meta


def _list_saved_runs_sync(settings: Any) -> list[dict]:
    saved_dir = settings.saved_runs_dir
    if not saved_dir.exists():
        return []
    metas = []
    for run_dir in sorted(saved_dir.iterdir()):
        meta_path = run_dir / "metadata.json"
        if meta_path.exists():
            with open(meta_path) as f:
                metas.append(json.load(f))
    return metas


def _get_saved_run_sync(run_id: str, settings: Any) -> dict | None:
    result_path = settings.saved_runs_dir / run_id / "result.json.gz"
    if not result_path.exists():
        return None
    with gzip.open(result_path, "rt", encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Main sync worker
# ---------------------------------------------------------------------------

def _run_price_forecast_sync(req: PriceForecastRequest) -> dict[str, Any]:
    from core.backtest.data.loader import load_assets, resample_ohlcv

    settings = get_settings()
    data_dir = req.data_dir if req.data_dir else str(settings.data_dir)

    horizon_alias = HORIZON_MAP.get(req.forecast_horizon, req.forecast_horizon)
    warn_list: list[str] = []

    # Determine which asset symbols to load
    assets_to_load = list(req.assets)
    if req.cross_asset and len(req.assets) == 2:
        pass  # both already in req.assets
    elif req.cross_asset and len(req.assets) == 1:
        other = "ETH" if req.assets[0] == "BTC" else "BTC"
        assets_to_load.append(other)
    if (
        req.regime_conditioning
        and req.regime_classifier_type == "threshold"
        and not ("BTC" in assets_to_load and "ETH" in assets_to_load)
    ):
        for a in ("BTC", "ETH"):
            if a not in assets_to_load:
                assets_to_load.append(a)

    symbols_to_load = {k: v for k, v in req.asset_symbols.items() if k in assets_to_load}
    raw_1min_multi = load_assets(data_dir, symbols_to_load, freq="1min")

    per_asset: dict[str, Any] = {}
    fitted_models: dict[str, dict] = {}
    # Keep raw signal pd.Series for regime conditioning
    raw_signals_aligned: dict[str, dict[str, pd.Series]] = {}
    raw_next_ret: dict[str, pd.Series] = {}

    for asset in req.assets:
        asset_ohlcv_1min: pd.DataFrame = raw_1min_multi[asset]
        horizon_ohlcv = resample_ohlcv(asset_ohlcv_1min, horizon_alias)
        close_h = horizon_ohlcv["close"]

        # Feature 1 — date split at horizon level
        _close_train_h, close_test_h, next_ret_test = _apply_date_split(close_h, req)

        other_asset = "ETH" if asset == "BTC" else "BTC"
        signals: dict[str, pd.Series] = {}
        model_resolutions_used: dict[str, str] = {}
        asset_fitted: dict[str, Any] = {}

        # Build model list (possibly including cross-asset models)
        models_for_asset = list(req.models)
        if req.cross_asset and len(req.assets) == 2:
            if "cross_tsmom" not in models_for_asset:
                models_for_asset.append("cross_tsmom")
            if "cross_momentum" not in models_for_asset:
                models_for_asset.append("cross_momentum")

        for model_name in models_for_asset:
            res = req.model_resolutions.get(model_name, "1D")
            is_cross = model_name.startswith("cross_")

            if is_cross:
                if other_asset not in raw_1min_multi:
                    warn_list.append(f"{model_name} skipped: {other_asset} data not loaded")
                    continue
                other_1min = raw_1min_multi[other_asset]
                other_model_ohlcv = resample_ohlcv(other_1min, res)
                other_close_res = other_model_ohlcv["close"]
                if req.train_start:
                    other_train_res = other_close_res[req.train_start : req.train_end]
                    other_test_res = other_close_res[req.test_start : req.test_end]
                else:
                    other_train_res = other_close_res
                    other_test_res = other_close_res
                if model_name == "cross_hp_filter" and res in RESOLUTION_FINE and len(other_test_res) > 500:
                    warn_list.append(
                        f"cross_hp_filter skipped at {res} (n={len(other_test_res)} > 500, O(n²) cost)"
                    )
                    continue
                if len(other_train_res) == 0:
                    warn_list.append(
                        f"{model_name} skipped: train slice is empty "
                        f"({req.train_start} → {req.train_end})"
                    )
                    continue
                if len(other_test_res) == 0:
                    warn_list.append(
                        f"{model_name} skipped: test slice is empty "
                        f"({req.test_start} → {req.test_end})"
                    )
                    continue
                m = _get_model(model_name)
                if m is None:
                    warn_list.append(f"Unknown model: {model_name}")
                    continue
                m.fit(other_train_res)
                signals[model_name] = m.predict(other_test_res)
                model_resolutions_used[model_name] = res
                asset_fitted[model_name] = m

            else:
                model_ohlcv = resample_ohlcv(asset_ohlcv_1min, res)
                close_at_res = model_ohlcv["close"]
                if req.train_start:
                    close_train_res = close_at_res[req.train_start : req.train_end]
                    close_test_res = close_at_res[req.test_start : req.test_end]
                else:
                    close_train_res = close_at_res
                    close_test_res = close_at_res
                if model_name == "hp_filter" and res in RESOLUTION_FINE and len(close_test_res) > 500:
                    warn_list.append(
                        f"hp_filter skipped at {res} (n={len(close_test_res)} > 500, O(n²) cost)"
                    )
                    continue
                if len(close_train_res) == 0:
                    warn_list.append(
                        f"{model_name} skipped: train slice is empty "
                        f"({req.train_start} → {req.train_end})"
                    )
                    continue
                if len(close_test_res) == 0:
                    warn_list.append(
                        f"{model_name} skipped: test slice is empty "
                        f"({req.test_start} → {req.test_end})"
                    )
                    continue
                m = _get_model(model_name)
                if m is None:
                    warn_list.append(f"Unknown model: {model_name}")
                    continue
                m.fit(close_train_res)
                signals[model_name] = m.predict(close_test_res)
                model_resolutions_used[model_name] = res
                asset_fitted[model_name] = m

        # Compute metrics
        metrics: dict[str, Any] = {}
        sig_aligned_map: dict[str, pd.Series] = {}
        for model_name, sig in signals.items():
            res = model_resolutions_used[model_name]
            sig_aligned = _align(sig, res, horizon_alias)
            output_type = OUTPUT_TYPES.get(model_name, "amplitude")
            metrics[model_name] = _compute_metrics_for_slice(
                sig_aligned, next_ret_test, req.n_calibration_bins, output_type, sig
            )
            sig_aligned_map[model_name] = sig_aligned

        per_asset[asset] = {
            "signals": {k: _series_to_timeseries(v) for k, v in signals.items()},
            "metrics": metrics,
            "prices": _series_to_timeseries(close_test_h),
            "model_resolutions": model_resolutions_used,
        }
        fitted_models[asset] = asset_fitted
        raw_signals_aligned[asset] = sig_aligned_map
        raw_next_ret[asset] = next_ret_test

    # Feature 4 — Regime conditioning
    regime_labels: list | None = None
    regime_metrics: dict | None = None

    if req.regime_conditioning:
        regime_series: pd.Series | None = None

        if req.regime_classifier_type == "vol_quantile":
            n_regimes = req.regime_classifier_params.get("n_regimes", 2)
            vol_window = req.regime_classifier_params.get("vol_window", 21)
            primary_asset = req.assets[0]
            # Full data range — quantile thresholds estimated on all available bars
            primary_h = resample_ohlcv(raw_1min_multi[primary_asset], horizon_alias)["close"]
            regime_series = _classify_vol_quantile(primary_h, n_regimes, vol_window)

        elif req.regime_classifier_type == "threshold":
            if "BTC" not in raw_1min_multi or "ETH" not in raw_1min_multi:
                warn_list.append("threshold classifier requires both BTC and ETH data")
            else:
                # Full data range — O(n²), warn if large
                btc_h = resample_ohlcv(raw_1min_multi["BTC"], horizon_alias)["close"]
                eth_h = resample_ohlcv(raw_1min_multi["ETH"], horizon_alias)["close"]
                if len(btc_h) > 500:
                    warn_list.append(
                        f"threshold classifier on {len(btc_h)} bars — may be slow (O(n²) ADF)"
                    )
                try:
                    regime_series = _classify_threshold(btc_h, eth_h, req.regime_classifier_params)
                except Exception as e:
                    warn_list.append(f"threshold classifier failed: {e}")

        elif req.regime_classifier_type == "manual":
            date_ranges = req.regime_classifier_params.get("date_ranges", [])
            primary_asset = req.assets[0]
            # Full data range — user-provided date ranges can span train + test
            primary_h = resample_ohlcv(raw_1min_multi[primary_asset], horizon_alias)["close"]
            regime_series = _classify_manual(primary_h.index, date_ranges)

        if regime_series is not None:
            valid_regime = regime_series.dropna()
            if len(valid_regime) > 0:
                unique_labels = sorted(str(lb) for lb in valid_regime.unique())
                regime_labels = [[ts.isoformat(), str(lb)] for ts, lb in valid_regime.items()]

                regime_metrics = {}
                for asset in req.assets:
                    regime_metrics[asset] = {}
                    for model_name, sig_aligned in raw_signals_aligned[asset].items():
                        output_type = OUTPUT_TYPES.get(model_name, "amplitude")
                        next_ret = raw_next_ret[asset]
                        regime_metrics[asset][model_name] = {}
                        for label in unique_labels:
                            label_mask = regime_series == label
                            common_idx = (
                                sig_aligned.dropna().index
                                .intersection(next_ret.dropna().index)
                                .intersection(label_mask[label_mask].index)
                            )
                            if len(common_idx) < 10:
                                warn_list.append(
                                    f"Regime '{label}' {asset}/{model_name}: "
                                    f"only {len(common_idx)} bars, skipped"
                                )
                                continue
                            regime_metrics[asset][model_name][label] = _compute_metrics_for_slice(
                                sig_aligned.loc[common_idx],
                                next_ret.loc[common_idx],
                                req.n_calibration_bins,
                                output_type,
                            )

    result: dict[str, Any] = {
        "assets": req.assets,
        "per_asset": per_asset,
        "forecast_horizon": req.forecast_horizon,
        "warnings": warn_list,
        "train_period": [req.train_start, req.train_end],
        "test_period": [req.test_start, req.test_end],
        "_fitted_models": fitted_models,
    }
    if regime_labels is not None:
        result["regime_labels"] = regime_labels
    if regime_metrics is not None:
        result["regime_metrics"] = regime_metrics

    return result


# ---------------------------------------------------------------------------
# Endpoints  (order matters: static paths before /{job_id})
# ---------------------------------------------------------------------------

def _get_data_range_sync(data_dir: str, asset_symbols: dict[str, str]) -> dict[str, Any]:
    from core.backtest.data.loader import load_assets, resample_ohlcv

    try:
        raw = load_assets(data_dir, asset_symbols, freq="1min")
    except Exception as e:
        return {"error": str(e)}

    result: dict[str, Any] = {}
    for asset_name, ohlcv in raw.items():
        try:
            close = resample_ohlcv(ohlcv, "1D")["close"].dropna()
            result[asset_name] = {
                "min": close.index[0].strftime("%Y-%m-%d"),
                "max": close.index[-1].strftime("%Y-%m-%d"),
                "total_bars": len(close),
            }
        except Exception:
            pass
    return result


@router.get("/data-range")
async def get_data_range(request: Request) -> dict:
    """Return available date range for each asset (used to constrain date pickers)."""
    settings = get_settings()
    from back_api.schemas.price_forecast import PriceForecastRequest as _Req
    default_symbols = _Req().asset_symbols
    return await asyncio.to_thread(
        _get_data_range_sync, str(settings.data_dir), default_symbols
    )


@router.post("/run")
async def run_price_forecast(req: PriceForecastRequest, request: Request) -> dict[str, str]:
    """Submit a price forecast evaluation job."""
    store: JobStore = _get_store(request)
    job_id = store.create()
    asyncio.create_task(
        store.run(job_id, asyncio.to_thread(_run_price_forecast_sync, req))
    )
    return {"job_id": job_id}


@router.post("/{job_id}/save")
async def save_price_forecast_run(
    job_id: str, req: SaveRunRequest, request: Request
) -> dict:
    """Persist a completed price-forecast run to disk and return metadata."""
    store: JobStore = _get_store(request)
    rec = store.get(job_id)
    if rec is None:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found.")
    if rec.status != "done":
        raise HTTPException(
            status_code=400,
            detail=f"Job '{job_id}' is not done (status: {rec.status}).",
        )
    run_id = str(uuid.uuid4())
    settings = get_settings()
    # Shallow copy so _save_run_sync can pop _fitted_models without affecting the store
    result_copy: dict[str, Any] = dict(rec.result or {})
    meta = await asyncio.to_thread(_save_run_sync, run_id, req.run_name, result_copy, settings)
    return meta.model_dump()


@router.get("/saved")
async def list_saved_runs(request: Request) -> list[dict]:
    """List all persisted price-forecast runs."""
    settings = get_settings()
    return await asyncio.to_thread(_list_saved_runs_sync, settings)


@router.get("/saved/{run_id}")
async def get_saved_run(run_id: str, request: Request) -> dict:
    """Load the full result of a persisted run."""
    settings = get_settings()
    result = await asyncio.to_thread(_get_saved_run_sync, run_id, settings)
    if result is None:
        raise HTTPException(status_code=404, detail=f"Saved run '{run_id}' not found.")
    return result


@router.get("/{job_id}", response_model=EvalJobStatusResponse)
async def get_price_forecast_status(job_id: str, request: Request) -> EvalJobStatusResponse:
    """Poll price forecast job status."""
    store: JobStore = _get_store(request)
    rec = store.get(job_id)
    if rec is None:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found.")
    result = rec.result
    if result is not None:
        # Strip private (non-JSON-serializable) keys before response
        result = {k: v for k, v in result.items() if not k.startswith("_")}
    return EvalJobStatusResponse(
        job_id=job_id,
        status=rec.status,
        result=result if rec.status == "done" else None,
        error=rec.error,
    )
