from __future__ import annotations
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

import polars as pl

from core.events import MarketEvent


def stream_events(parquet_path: Path, symbol: str, funding_path: Path | None = None) -> Iterator[MarketEvent]:
    df = pl.read_parquet(parquet_path)

    funding: dict[int, float] = {}
    if funding_path and funding_path.exists():
        fdf = pl.read_parquet(funding_path)
        col = next((c for c in fdf.columns if "time" in c.lower()), fdf.columns[0])
        rate_col = next((c for c in fdf.columns if "rate" in c.lower()), fdf.columns[1])
        for row in fdf.iter_rows(named=True):
            funding[int(row[col]) // 1000] = float(row[rate_col])

    for row in df.iter_rows(named=True):
        ts = row["timestamp"]
        dt = datetime.fromtimestamp(ts, tz=timezone.utc).replace(tzinfo=None)
        yield MarketEvent(
            timestamp=dt,
            symbol=symbol,
            open=row["open"],
            high=row["high"],
            low=row["low"],
            close=row["close"],
            volume=row["volume"],
            funding_rate=funding.get(ts),
        )
