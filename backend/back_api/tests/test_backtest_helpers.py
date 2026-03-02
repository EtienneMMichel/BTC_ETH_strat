"""Unit tests for _request_fingerprint and _data_cache_key."""
from __future__ import annotations

import json

import pytest

from back_api.routers.backtest import _data_cache_key, _request_fingerprint
from back_api.schemas.backtest import BacktestRequest


def _req(**overrides) -> BacktestRequest:
    defaults = dict(
        assets={"BTC": "BTCUSDT", "ETH": "ETHUSDT"},
        freq="1D",
        strategy="orchestrator",
    )
    defaults.update(overrides)
    return BacktestRequest(**defaults)


# ---------------------------------------------------------------------------
# _request_fingerprint
# ---------------------------------------------------------------------------

class TestRequestFingerprint:
    def test_same_request_same_fingerprint(self):
        req = _req()
        assert _request_fingerprint(req) == _request_fingerprint(req)

    def test_different_strategy_different_fingerprint(self):
        assert _request_fingerprint(_req(strategy="momentum")) != _request_fingerprint(
            _req(strategy="mean_reversion")
        )

    def test_asset_order_does_not_matter(self):
        fp1 = _request_fingerprint(_req(assets={"BTC": "BTCUSDT", "ETH": "ETHUSDT"}))
        fp2 = _request_fingerprint(_req(assets={"ETH": "ETHUSDT", "BTC": "BTCUSDT"}))
        assert fp1 == fp2

    def test_different_freq_different_fingerprint(self):
        assert _request_fingerprint(_req(freq="1D")) != _request_fingerprint(_req(freq="4h"))

    def test_different_assets_different_fingerprint(self):
        fp1 = _request_fingerprint(_req(assets={"BTC": "BTCUSDT"}))
        fp2 = _request_fingerprint(_req(assets={"ETH": "ETHUSDT"}))
        assert fp1 != fp2

    def test_returns_valid_json_string(self):
        fp = _request_fingerprint(_req())
        assert isinstance(fp, str)
        parsed = json.loads(fp)
        assert isinstance(parsed, dict)

    def test_numeric_params_affect_fingerprint(self):
        fp1 = _request_fingerprint(_req(min_train_bars=252))
        fp2 = _request_fingerprint(_req(min_train_bars=100))
        assert fp1 != fp2


# ---------------------------------------------------------------------------
# _data_cache_key
# ---------------------------------------------------------------------------

class TestDataCacheKey:
    def test_asset_order_does_not_matter(self):
        k1 = _data_cache_key("/data", {"BTC": "BTCUSDT", "ETH": "ETHUSDT"}, "1D")
        k2 = _data_cache_key("/data", {"ETH": "ETHUSDT", "BTC": "BTCUSDT"}, "1D")
        assert k1 == k2

    def test_different_dir_different_key(self):
        k1 = _data_cache_key("/data/a", {"BTC": "BTCUSDT"}, "1D")
        k2 = _data_cache_key("/data/b", {"BTC": "BTCUSDT"}, "1D")
        assert k1 != k2

    def test_different_freq_different_key(self):
        k1 = _data_cache_key("/data", {"BTC": "BTCUSDT"}, "1D")
        k2 = _data_cache_key("/data", {"BTC": "BTCUSDT"}, "4h")
        assert k1 != k2

    def test_different_assets_different_key(self):
        k1 = _data_cache_key("/data", {"BTC": "BTCUSDT"}, "1D")
        k2 = _data_cache_key("/data", {"ETH": "ETHUSDT"}, "1D")
        assert k1 != k2

    def test_is_hashable(self):
        k = _data_cache_key("/data", {"BTC": "BTCUSDT", "ETH": "ETHUSDT"}, "1D")
        d = {k: "ok"}
        assert d[k] == "ok"
