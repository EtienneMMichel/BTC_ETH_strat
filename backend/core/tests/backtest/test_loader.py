import tempfile
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from core.backtest.data.loader import load_ohlcv, resample_ohlcv, load_assets


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


# ---------------------------------------------------------------------------
# resample_ohlcv tests
# ---------------------------------------------------------------------------

def make_1min_ohlcv(n_days=5):
    """Create synthetic 1-minute OHLCV bars for n_days."""
    rng = np.random.default_rng(42)
    n = n_days * 24 * 60
    idx = pd.date_range("2022-01-01", periods=n, freq="1min", tz="UTC")
    close = 10000 * np.exp(np.cumsum(rng.standard_normal(n) * 0.0001))
    return pd.DataFrame({
        "open":   close * (1 + rng.standard_normal(n) * 0.0001),
        "high":   close * (1 + np.abs(rng.standard_normal(n)) * 0.0002),
        "low":    close * (1 - np.abs(rng.standard_normal(n)) * 0.0002),
        "close":  close,
        "volume": rng.uniform(1, 10, n),
    }, index=idx)


def test_resample_daily_length():
    df = make_1min_ohlcv(n_days=5)
    daily = resample_ohlcv(df, "1D")
    assert len(daily) == 5


def test_resample_hourly_length():
    df = make_1min_ohlcv(n_days=2)
    hourly = resample_ohlcv(df, "1h")
    assert len(hourly) == 48


def test_resample_high_is_max():
    df = make_1min_ohlcv(n_days=3)
    daily = resample_ohlcv(df, "1D")
    for date, group in df.groupby(df.index.date):
        expected_high = group["high"].max()
        row = daily[daily.index.date == date]
        if len(row):
            assert row["high"].iloc[0] == pytest.approx(expected_high)


def test_resample_volume_is_sum():
    df = make_1min_ohlcv(n_days=3)
    daily = resample_ohlcv(df, "1D")
    for date, group in df.groupby(df.index.date):
        expected_vol = group["volume"].sum()
        row = daily[daily.index.date == date]
        if len(row):
            assert row["volume"].iloc[0] == pytest.approx(expected_vol)


def test_resample_open_is_first():
    df = make_1min_ohlcv(n_days=3)
    daily = resample_ohlcv(df, "1D")
    for date, group in df.groupby(df.index.date):
        expected_open = group["open"].iloc[0]
        row = daily[daily.index.date == date]
        if len(row):
            assert row["open"].iloc[0] == pytest.approx(expected_open)


def test_resample_close_is_last():
    df = make_1min_ohlcv(n_days=3)
    daily = resample_ohlcv(df, "1D")
    for date, group in df.groupby(df.index.date):
        expected_close = group["close"].iloc[-1]
        row = daily[daily.index.date == date]
        if len(row):
            assert row["close"].iloc[0] == pytest.approx(expected_close)


def test_resample_utc_index():
    df = make_1min_ohlcv(n_days=3)
    daily = resample_ohlcv(df, "1D")
    assert daily.index.tz is not None


# ---------------------------------------------------------------------------
# load_assets tests
# ---------------------------------------------------------------------------

def make_raw_parquet(tmp_path: Path, symbol: str, year: int, month: int,
                     n_days: int = 3) -> Path:
    """Write a Binance-format 1m parquet file (unix-seconds timestamp column)."""
    sym_dir = tmp_path / symbol
    sym_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(hash((symbol, year, month)) % (2**31))
    n = n_days * 24 * 60
    start = pd.Timestamp(f"{year}-{month:02d}-01", tz="UTC")
    timestamps = pd.date_range(start, periods=n, freq="1min")
    close = 10000 * np.exp(np.cumsum(rng.standard_normal(n) * 0.0001))
    df = pd.DataFrame({
        "timestamp": timestamps.tz_localize(None).astype("datetime64[s]").astype("int64"),  # unix seconds
        "open":   close * 0.9999,
        "high":   close * 1.0002,
        "low":    close * 0.9998,
        "close":  close,
        "volume": rng.uniform(1, 10, n),
    })
    out = sym_dir / f"{symbol}_1m_{year}_{month:02d}.parquet"
    df.to_parquet(out, index=False)
    return out


def test_load_assets_multiindex(tmp_path):
    make_raw_parquet(tmp_path, "BTCUSDT", 2022, 1)
    make_raw_parquet(tmp_path, "ETHUSDT", 2022, 1)
    df = load_assets(tmp_path, {"BTC": "BTCUSDT", "ETH": "ETHUSDT"}, freq="1D")
    assert isinstance(df.columns, pd.MultiIndex)
    assert ("BTC", "close") in df.columns
    assert ("ETH", "close") in df.columns


def test_load_assets_freq_daily(tmp_path):
    make_raw_parquet(tmp_path, "BTCUSDT", 2022, 1, n_days=5)
    make_raw_parquet(tmp_path, "ETHUSDT", 2022, 1, n_days=5)
    df = load_assets(tmp_path, {"BTC": "BTCUSDT", "ETH": "ETHUSDT"}, freq="1D")
    assert len(df) == 5


def test_load_assets_freq_hourly(tmp_path):
    make_raw_parquet(tmp_path, "BTCUSDT", 2022, 1, n_days=2)
    make_raw_parquet(tmp_path, "ETHUSDT", 2022, 1, n_days=2)
    df = load_assets(tmp_path, {"BTC": "BTCUSDT", "ETH": "ETHUSDT"}, freq="1h")
    assert len(df) == 48


def test_load_assets_year_filter(tmp_path):
    make_raw_parquet(tmp_path, "BTCUSDT", 2022, 1, n_days=3)
    make_raw_parquet(tmp_path, "BTCUSDT", 2023, 1, n_days=3)
    make_raw_parquet(tmp_path, "ETHUSDT", 2022, 1, n_days=3)
    make_raw_parquet(tmp_path, "ETHUSDT", 2023, 1, n_days=3)
    df_22 = load_assets(tmp_path, {"BTC": "BTCUSDT", "ETH": "ETHUSDT"},
                        freq="1D", years=[2022])
    df_both = load_assets(tmp_path, {"BTC": "BTCUSDT", "ETH": "ETHUSDT"},
                          freq="1D", years=[2022, 2023])
    assert len(df_22) < len(df_both)


def test_load_assets_missing_symbol_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_assets(tmp_path, {"BTC": "BTCUSDT"}, freq="1D")
