import tempfile
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from core.backtest.data.loader import load_ohlcv


def write_csv(tmp_path: Path, df: pd.DataFrame, filename="prices.csv") -> Path:
    p = tmp_path / filename
    df.to_csv(p)
    return p


def make_single_asset_csv(tmp_path, n=50, asset="BTC"):
    idx = pd.date_range("2022-01-01", periods=n, freq="D", tz="UTC")
    rng = np.random.default_rng(0)
    close = 10000 * np.exp(np.cumsum(rng.standard_normal(n) * 0.01))
    df = pd.DataFrame({
        "open": close * 0.99,
        "high": close * 1.01,
        "low": close * 0.98,
        "close": close,
        "volume": rng.uniform(100, 500, n),
    }, index=idx)
    return write_csv(tmp_path, df), df


def test_load_returns_dataframe(tmp_path):
    path, _ = make_single_asset_csv(tmp_path)
    df = load_ohlcv(path, assets=["BTC"])
    assert isinstance(df, pd.DataFrame)


def test_load_multiindex_columns(tmp_path):
    path, _ = make_single_asset_csv(tmp_path)
    df = load_ohlcv(path, assets=["BTC"])
    assert isinstance(df.columns, pd.MultiIndex)


def test_load_utc_index(tmp_path):
    path, _ = make_single_asset_csv(tmp_path)
    df = load_ohlcv(path, assets=["BTC"])
    assert df.index.tz is not None


def test_load_length(tmp_path):
    path, orig = make_single_asset_csv(tmp_path, n=60)
    df = load_ohlcv(path, assets=["BTC"])
    assert len(df) == 60


def test_duplicate_timestamps_raises(tmp_path):
    idx = pd.DatetimeIndex(
        ["2022-01-01", "2022-01-01", "2022-01-02"], tz="UTC"
    )
    df = pd.DataFrame({"open": 1, "high": 1, "low": 1, "close": 1, "volume": 1}, index=idx)
    path = tmp_path / "dup.csv"
    df.to_csv(path)
    with pytest.raises(ValueError, match="Duplicate"):
        load_ohlcv(path, assets=["BTC"])


def test_non_monotonic_raises(tmp_path):
    idx = pd.DatetimeIndex(["2022-01-03", "2022-01-01", "2022-01-02"], tz="UTC")
    df = pd.DataFrame({"open": 1, "high": 1, "low": 1, "close": 1, "volume": 1}, index=idx)
    path = tmp_path / "nonmono.csv"
    df.to_csv(path)
    with pytest.raises(ValueError, match="monotonic"):
        load_ohlcv(path, assets=["BTC"])
