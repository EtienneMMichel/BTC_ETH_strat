"""Integration and unit tests for the backtest router and its three cache layers."""
from __future__ import annotations

import asyncio
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from back_api.routers.backtest import (
    _DATA_CACHE,
    _RESULT_CACHE,
    _request_fingerprint,
    _run_backtest_sync,
)
from back_api.schemas.backtest import BacktestRequest, BacktestResultSchema

# ---------------------------------------------------------------------------
# Shared test data
# ---------------------------------------------------------------------------

BASE_REQUEST: dict = {
    "assets": {"BTC": "BTCUSDT", "ETH": "ETHUSDT"},
    "freq": "1D",
    "strategy": "orchestrator",
}

FAKE_RESULT = BacktestResultSchema(
    equity_curve=[
        ("2024-01-01T00:00:00+00:00", 1.0),
        ("2024-01-02T00:00:00+00:00", 1.02),
    ],
    metrics={
        "sharpe": 1.5,
        "max_drawdown": -0.05,
        "calmar": 30.0,
        "var": -0.01,
        "es": -0.015,
        "win_rate": 0.6,
    },
    trade_count=2,
    signals=[],
)


def _make_fake_ohlcv() -> pd.DataFrame:
    """Minimal MultiIndex OHLCV DataFrame compatible with load_assets output."""
    idx = pd.date_range("2020-01-01", periods=300, freq="D", tz="UTC")
    prices = list(range(1, 301))
    return pd.concat(
        {
            "BTC": pd.DataFrame({"close": prices, "open": prices}, index=idx),
            "ETH": pd.DataFrame({"close": prices, "open": prices}, index=idx),
        },
        axis=1,
    )


async def _wait_for_done(
    client, prefix: str, job_id: str, max_wait: float = 3.0
) -> dict | None:
    elapsed = 0.0
    while elapsed < max_wait:
        resp = await client.get(f"{prefix}/{job_id}")
        data = resp.json()
        if data["status"] in ("done", "failed"):
            return data
        await asyncio.sleep(0.05)
        elapsed += 0.05
    return None


# ---------------------------------------------------------------------------
# Health endpoint
# ---------------------------------------------------------------------------

async def test_health(api_client):
    resp = await api_client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


# ---------------------------------------------------------------------------
# Submit + poll
# ---------------------------------------------------------------------------

async def test_submit_returns_job_id(api_client):
    with patch(
        "back_api.routers.backtest._run_backtest_sync", return_value=FAKE_RESULT
    ):
        resp = await api_client.post("/backtest/run", json=BASE_REQUEST)
    assert resp.status_code == 200
    assert "job_id" in resp.json()
    assert isinstance(resp.json()["job_id"], str)


async def test_get_unknown_job_returns_404(api_client):
    resp = await api_client.get("/backtest/not-a-real-id")
    assert resp.status_code == 404


async def test_get_status_returns_job_id_field(api_client):
    with patch(
        "back_api.routers.backtest._run_backtest_sync", return_value=FAKE_RESULT
    ):
        resp = await api_client.post("/backtest/run", json=BASE_REQUEST)
        job_id = resp.json()["job_id"]
        status = await api_client.get(f"/backtest/{job_id}")
    assert status.status_code == 200
    assert status.json()["job_id"] == job_id


async def test_completed_job_has_result(api_client):
    with patch(
        "back_api.routers.backtest._run_backtest_sync", return_value=FAKE_RESULT
    ):
        resp = await api_client.post("/backtest/run", json=BASE_REQUEST)
        job_id = resp.json()["job_id"]
        data = await _wait_for_done(api_client, "/backtest", job_id)

    assert data is not None
    assert data["status"] == "done"
    result = data["result"]
    assert "equity_curve" in result
    assert "metrics" in result
    assert "trade_count" in result


async def test_failed_job_has_error(api_client):
    with patch(
        "back_api.routers.backtest._run_backtest_sync",
        side_effect=RuntimeError("engine exploded"),
    ):
        resp = await api_client.post("/backtest/run", json=BASE_REQUEST)
        job_id = resp.json()["job_id"]
        data = await _wait_for_done(api_client, "/backtest", job_id)

    assert data is not None
    assert data["status"] == "failed"
    assert "engine exploded" in data["error"]
    assert data["result"] is None


# ---------------------------------------------------------------------------
# Layer 3 — job deduplication
# ---------------------------------------------------------------------------

async def test_layer3_same_request_same_job_id(api_client):
    with patch(
        "back_api.routers.backtest._run_backtest_sync", return_value=FAKE_RESULT
    ):
        resp1 = await api_client.post("/backtest/run", json=BASE_REQUEST)
        resp2 = await api_client.post("/backtest/run", json=BASE_REQUEST)

    assert resp1.json()["job_id"] == resp2.json()["job_id"]


async def test_layer3_different_strategy_different_job_id(api_client):
    req_momentum = {**BASE_REQUEST, "strategy": "momentum"}
    with patch(
        "back_api.routers.backtest._run_backtest_sync", return_value=FAKE_RESULT
    ):
        resp1 = await api_client.post("/backtest/run", json=BASE_REQUEST)
        resp2 = await api_client.post("/backtest/run", json=req_momentum)

    assert resp1.json()["job_id"] != resp2.json()["job_id"]


async def test_layer3_failed_job_creates_new_job(api_client, fresh_store):
    """After a job fails its fingerprint slot is freed; the same request gets a new job."""
    with patch(
        "back_api.routers.backtest._run_backtest_sync",
        side_effect=RuntimeError("fail"),
    ):
        resp1 = await api_client.post("/backtest/run", json=BASE_REQUEST)
        job_id1 = resp1.json()["job_id"]
        await _wait_for_done(api_client, "/backtest", job_id1)

    # After failure, same request must create a new job
    with patch(
        "back_api.routers.backtest._run_backtest_sync", return_value=FAKE_RESULT
    ):
        resp2 = await api_client.post("/backtest/run", json=BASE_REQUEST)

    assert resp1.json()["job_id"] != resp2.json()["job_id"]


# ---------------------------------------------------------------------------
# Layer 2 — result cache
# ---------------------------------------------------------------------------

def test_layer2_cache_hit_skips_data_loading():
    """Pre-populating _RESULT_CACHE makes _run_backtest_sync return without
    ever calling load_assets."""
    req = BacktestRequest(**{**BASE_REQUEST, "strategy": "momentum"})
    fp = _request_fingerprint(req)
    _RESULT_CACHE[fp] = FAKE_RESULT

    with patch("core.backtest.data.loader.load_assets") as mock_load:
        result = _run_backtest_sync(req)

    assert result is FAKE_RESULT
    mock_load.assert_not_called()


def test_layer2_cache_populated_after_run():
    """After _run_backtest_sync completes, the result is stored in _RESULT_CACHE."""
    req = BacktestRequest(**{**BASE_REQUEST, "strategy": "momentum"})
    fp = _request_fingerprint(req)

    fake_ohlcv = _make_fake_ohlcv()
    mock_run_result = MagicMock()
    mock_run_result.equity_curve = {}
    mock_run_result.signals = pd.DataFrame()
    mock_run_result.trade_log = []
    mock_run_result.metrics = {}

    with (
        patch("core.backtest.data.loader.load_assets", return_value=fake_ohlcv),
        patch("core.backtest.engine.runner.WalkForwardRunner") as MockRunner,
        patch("core.strats.momentum.strategy.MomentumStrategy"),
    ):
        MockRunner.return_value.run.return_value = mock_run_result
        _run_backtest_sync(req)

    assert fp in _RESULT_CACHE


# ---------------------------------------------------------------------------
# Layer 1 — data cache
# ---------------------------------------------------------------------------

def test_layer1_data_cache_avoids_second_load_assets_call():
    """Two requests sharing the same data parameters (assets + freq) call
    load_assets exactly once regardless of how many times the engine runs."""
    req_mom = BacktestRequest(**{**BASE_REQUEST, "strategy": "momentum"})
    req_rev = BacktestRequest(**{**BASE_REQUEST, "strategy": "mean_reversion"})

    fake_ohlcv = _make_fake_ohlcv()
    mock_run_result = MagicMock()
    mock_run_result.equity_curve = {}
    mock_run_result.signals = pd.DataFrame()
    mock_run_result.trade_log = []
    mock_run_result.metrics = {}

    with (
        patch("core.backtest.data.loader.load_assets", return_value=fake_ohlcv) as mock_load,
        patch("core.backtest.engine.runner.WalkForwardRunner") as MockRunner,
        patch("core.strats.momentum.strategy.MomentumStrategy"),
        patch("core.strats.mean_reversion.strategy.MeanReversionStrategy"),
    ):
        MockRunner.return_value.run.return_value = mock_run_result
        _run_backtest_sync(req_mom)  # cold: loads data, stores in _DATA_CACHE
        _run_backtest_sync(req_rev)  # warm: skips load_assets

    assert mock_load.call_count == 1


def test_layer1_data_cache_populated_after_run():
    """After _run_backtest_sync completes, the ohlcv DataFrame is in _DATA_CACHE."""
    from back_api.routers.backtest import _data_cache_key
    from back_api.config import get_settings

    req = BacktestRequest(**BASE_REQUEST)
    settings = get_settings()
    data_dir = str(settings.data_dir)
    dk = _data_cache_key(data_dir, req.assets, req.freq)

    fake_ohlcv = _make_fake_ohlcv()
    mock_run_result = MagicMock()
    mock_run_result.equity_curve = {}
    mock_run_result.signals = pd.DataFrame()
    mock_run_result.trade_log = []
    mock_run_result.metrics = {}

    with (
        patch("core.backtest.data.loader.load_assets", return_value=fake_ohlcv),
        patch("core.backtest.engine.runner.WalkForwardRunner") as MockRunner,
        patch("core.strats.orchestrator.regime.OrchestratorStrategy"),
    ):
        MockRunner.return_value.run.return_value = mock_run_result
        _run_backtest_sync(req)

    assert dk in _DATA_CACHE
