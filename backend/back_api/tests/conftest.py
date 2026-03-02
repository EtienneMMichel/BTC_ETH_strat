from __future__ import annotations

import pytest
from httpx import ASGITransport, AsyncClient

import back_api.routers.backtest as backtest_module
from back_api.jobs import JobStore
from back_api.main import app


@pytest.fixture(autouse=True)
def clear_caches():
    """Clear module-level caches before/after each test for isolation."""
    backtest_module._DATA_CACHE.clear()
    backtest_module._RESULT_CACHE.clear()
    yield
    backtest_module._DATA_CACHE.clear()
    backtest_module._RESULT_CACHE.clear()


@pytest.fixture
def fresh_store() -> JobStore:
    return JobStore()


@pytest.fixture
async def api_client(fresh_store: JobStore):
    app.state.job_store = fresh_store
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as ac:
        yield ac
