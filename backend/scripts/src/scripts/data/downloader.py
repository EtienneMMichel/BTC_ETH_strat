from __future__ import annotations
import io
import zipfile
from pathlib import Path

import polars as pl
import requests

BASE_URL = "https://data.binance.vision/data/futures/um/monthly/klines"
FUNDING_URL = "https://data.binance.vision/data/futures/um/monthly/fundingRate"

CANDLE_COLS = ["open_time", "open", "high", "low", "close", "volume",
               "close_time", "quote_volume", "trades", "taker_base",
               "taker_quote", "ignore"]


def download_candles(symbol: str, year: int, month: int, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{symbol}_1m_{year}_{month:02d}.parquet"
    if out_path.exists():
        return out_path

    url = f"{BASE_URL}/{symbol}/1m/{symbol}-1m-{year}-{month:02d}.zip"
    resp = requests.get(url, timeout=60)
    resp.raise_for_status()

    with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
        csv_name = zf.namelist()[0]
        with zf.open(csv_name) as f:
            raw = f.read()

    # Newer Binance files include a header row; detect by checking whether
    # the first field is numeric (no header) or text (has header).
    has_header = not raw.split(b"\n")[0].split(b",")[0].strip(b'"').isdigit()
    df = pl.read_csv(
        io.BytesIO(raw),
        has_header=has_header,
        new_columns=None if has_header else CANDLE_COLS,
    )
    if has_header:
        df = df.rename(dict(zip(df.columns, CANDLE_COLS)))

    df = df.select([
        (pl.col("open_time").cast(pl.Int64) // 1000).alias("timestamp"),
        pl.col("open").cast(pl.Float64),
        pl.col("high").cast(pl.Float64),
        pl.col("low").cast(pl.Float64),
        pl.col("close").cast(pl.Float64),
        pl.col("volume").cast(pl.Float64),
    ])
    df.write_parquet(out_path)
    return out_path


def download_funding(symbol: str, year: int, month: int, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{symbol}_funding_{year}_{month:02d}.parquet"
    if out_path.exists():
        return out_path

    url = f"{FUNDING_URL}/{symbol}/{symbol}-fundingRate-{year}-{month:02d}.zip"
    resp = requests.get(url, timeout=60)
    resp.raise_for_status()

    with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
        with zf.open(zf.namelist()[0]) as f:
            df = pl.read_csv(f)

    df.write_parquet(out_path)
    return out_path
