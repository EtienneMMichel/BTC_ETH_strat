"""Integration tests for the evaluation router (vol-comparison, var-test, dm-test)."""
from __future__ import annotations

import asyncio
import datetime

import pytest

# ---------------------------------------------------------------------------
# Shared test data — 50 observations, non-constant to avoid degenerate stats
# ---------------------------------------------------------------------------

_DATES = [
    (datetime.date(2024, 1, 1) + datetime.timedelta(days=i)).isoformat()
    for i in range(50)
]

# Slightly varied series (% vol): avoids division-by-zero in QLIKE
_REALISED_VOL = [0.010 + 0.001 * (i % 5) for i in range(50)]
_FORECAST_GARCH = [0.010 + 0.0008 * (i % 5) for i in range(50)]
_FORECAST_EWMA = [0.012 + 0.001 * (i % 7) for i in range(50)]

# Returns / VaR for Kupiec + Christoffersen tests
_RETURNS = [-0.03] * 3 + [0.01] * 47          # 3 violations at -2% VaR
_VAR_SERIES = [-0.02] * 50

# DM test: two forecasts with non-zero loss differential
_REALISED_PRICE = [1.0 + 0.01 * (i % 7) for i in range(50)]
_FORECAST_A = [r * 0.99 for r in _REALISED_PRICE]
_FORECAST_B = [r * 1.02 for r in _REALISED_PRICE]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

async def _wait_for_done(client, job_id: str, max_wait: float = 5.0) -> dict | None:
    elapsed = 0.0
    while elapsed < max_wait:
        resp = await client.get(f"/evaluation/{job_id}")
        data = resp.json()
        if data["status"] in ("done", "failed"):
            return data
        await asyncio.sleep(0.1)
        elapsed += 0.1
    return None


# ---------------------------------------------------------------------------
# 404 for unknown job
# ---------------------------------------------------------------------------

async def test_unknown_job_returns_404(api_client):
    resp = await api_client.get("/evaluation/no-such-job")
    assert resp.status_code == 404


# ---------------------------------------------------------------------------
# Vol comparison
# ---------------------------------------------------------------------------

async def test_vol_comparison_submits_job(api_client):
    payload = {
        "forecasts": {"GARCH": _FORECAST_GARCH, "EWMA": _FORECAST_EWMA},
        "realised": _REALISED_VOL,
        "index": _DATES,
        "metrics": ["qlike", "mse", "mae"],
    }
    resp = await api_client.post("/evaluation/vol-comparison", json=payload)
    assert resp.status_code == 200
    assert "job_id" in resp.json()


async def test_vol_comparison_completes_with_model_keys(api_client):
    payload = {
        "forecasts": {"GARCH": _FORECAST_GARCH, "EWMA": _FORECAST_EWMA},
        "realised": _REALISED_VOL,
        "index": _DATES,
        "metrics": ["qlike", "mse", "mae"],
    }
    resp = await api_client.post("/evaluation/vol-comparison", json=payload)
    job_id = resp.json()["job_id"]

    data = await _wait_for_done(api_client, job_id)
    assert data is not None, "Job did not complete in time"
    assert data["status"] == "done"
    assert "GARCH" in data["result"]
    assert "EWMA" in data["result"]


# ---------------------------------------------------------------------------
# VaR test
# ---------------------------------------------------------------------------

async def test_var_test_submits_job(api_client):
    payload = {
        "returns": _RETURNS,
        "var_series": _VAR_SERIES,
        "alpha": 0.05,
        "index": _DATES,
    }
    resp = await api_client.post("/evaluation/var-test", json=payload)
    assert resp.status_code == 200
    assert "job_id" in resp.json()


async def test_var_test_completes_with_kupiec_and_christoffersen(api_client):
    payload = {
        "returns": _RETURNS,
        "var_series": _VAR_SERIES,
        "alpha": 0.05,
        "index": _DATES,
    }
    resp = await api_client.post("/evaluation/var-test", json=payload)
    job_id = resp.json()["job_id"]

    data = await _wait_for_done(api_client, job_id)
    assert data is not None, "Job did not complete in time"
    assert data["status"] == "done"
    assert "kupiec" in data["result"]
    assert "christoffersen" in data["result"]


# ---------------------------------------------------------------------------
# DM test
# ---------------------------------------------------------------------------

async def test_dm_test_submits_job(api_client):
    payload = {
        "forecast_a": _FORECAST_A,
        "forecast_b": _FORECAST_B,
        "realised": _REALISED_PRICE,
        "index": _DATES,
    }
    resp = await api_client.post("/evaluation/dm-test", json=payload)
    assert resp.status_code == 200
    assert "job_id" in resp.json()


async def test_dm_test_completes_with_stat_and_pvalue(api_client):
    payload = {
        "forecast_a": _FORECAST_A,
        "forecast_b": _FORECAST_B,
        "realised": _REALISED_PRICE,
        "index": _DATES,
        "loss": "mse",
        "h": 1,
    }
    resp = await api_client.post("/evaluation/dm-test", json=payload)
    job_id = resp.json()["job_id"]

    data = await _wait_for_done(api_client, job_id)
    assert data is not None, "Job did not complete in time"
    assert data["status"] == "done"
    assert "dm_stat" in data["result"]
    assert "p_value" in data["result"]


# ---------------------------------------------------------------------------
# Status polling
# ---------------------------------------------------------------------------

async def test_status_is_valid_after_submit(api_client):
    payload = {
        "forecast_a": _FORECAST_A,
        "forecast_b": _FORECAST_B,
        "realised": _REALISED_PRICE,
        "index": _DATES,
    }
    resp = await api_client.post("/evaluation/dm-test", json=payload)
    job_id = resp.json()["job_id"]

    status_resp = await api_client.get(f"/evaluation/{job_id}")
    assert status_resp.status_code == 200
    assert status_resp.json()["status"] in ("pending", "running", "done")
    assert status_resp.json()["job_id"] == job_id
