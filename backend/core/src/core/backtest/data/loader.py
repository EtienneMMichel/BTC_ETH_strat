from pathlib import Path
import pandas as pd


_OHLCV_AGG = {
    "open": "first",
    "high": "max",
    "low": "min",
    "close": "last",
    "volume": "sum",
}


def resample_ohlcv(df: pd.DataFrame, freq: str) -> pd.DataFrame:
    """
    Resample a single-asset OHLCV DataFrame to a new frequency.

    df must have a UTC DatetimeIndex and columns open, high, low, close, volume
    (flat, not MultiIndex).  freq is any pandas offset string: '1D', '4h', '1h',
    '15min', 'W', etc.

    Returns a resampled DataFrame with the same column set, dropping incomplete
    leading/trailing bars that contain NaN.
    """
    agg = {c: _OHLCV_AGG[c] for c in _OHLCV_AGG if c in df.columns}
    resampled = df.resample(freq).agg(agg).dropna()
    return resampled


def _load_raw_single(path: Path) -> pd.DataFrame:
    """
    Load one Binance raw 1m parquet file (timestamp as unix seconds integer).
    Returns a DataFrame with a UTC DatetimeIndex and flat OHLCV columns.
    """
    df = pd.read_parquet(path)
    # Convert unix-seconds timestamp column to DatetimeIndex
    if "timestamp" in df.columns:
        df.index = pd.to_datetime(df["timestamp"], unit="s", utc=True)
        df = df.drop(columns=["timestamp"])
    elif not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index, utc=True)
    elif df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    df.columns = [c.lower() for c in df.columns]
    return df


def load_assets(
    raw_dir: str | Path,
    assets: dict[str, str],
    freq: str = "1D",
    years: list[int] | None = None,
    months: list[int] | None = None,
) -> pd.DataFrame:
    """
    Load multiple assets from Binance raw 1m parquet files, resample, and
    return a MultiIndex-column DataFrame ready for the pipeline.

    Parameters
    ----------
    raw_dir : root directory containing one sub-folder per symbol
              e.g. data/raw/BTCUSDT/, data/raw/ETHUSDT/
    assets  : mapping from pipeline name to Binance symbol
              e.g. {"BTC": "BTCUSDT", "ETH": "ETHUSDT"}
    freq    : pandas offset string for resampling — "1D", "4h", "1h", "15min" …
    years   : filter to these years (None = all available)
    months  : filter to these months 1–12 (None = all available)

    Returns
    -------
    pd.DataFrame with pd.MultiIndex columns (asset, field) and UTC DatetimeIndex,
    aligned on the common date range across all assets.
    """
    raw_dir = Path(raw_dir)
    frames: dict[str, pd.DataFrame] = {}

    for name, symbol in assets.items():
        print("raw_dir: ", raw_dir)
        sym_dir = raw_dir / symbol
        if not sym_dir.exists():
            raise FileNotFoundError(f"Symbol directory not found: {sym_dir}")

        parts = []
        for p in sorted(sym_dir.glob(f"{symbol}_1m_*.parquet")):
            # filename: BTCUSDT_1m_2022_01.parquet
            stem_parts = p.stem.split("_")
            if len(stem_parts) < 4:
                continue
            try:
                yr, mo = int(stem_parts[-2]), int(stem_parts[-1])
            except ValueError:
                continue
            if years is not None and yr not in years:
                continue
            if months is not None and mo not in months:
                continue
            parts.append(_load_raw_single(p))

        if not parts:
            raise FileNotFoundError(
                f"No 1m parquet files found for {symbol} in {sym_dir} "
                f"(years={years}, months={months})"
            )

        raw = pd.concat(parts).sort_index()
        raw = raw[~raw.index.duplicated(keep="first")]
        frames[name] = resample_ohlcv(raw, freq)

    # Align on common index
    common_idx = frames[next(iter(frames))].index
    for f in frames.values():
        common_idx = common_idx.intersection(f.index)

    tuples = [(name, field) for name in assets for field in _OHLCV_AGG]
    arrays = [frames[name].loc[common_idx, field]
              for name in assets for field in _OHLCV_AGG]

    result = pd.concat(arrays, axis=1)
    result.columns = pd.MultiIndex.from_tuples(tuples)
    return result


def load_ohlcv(
    path: str | Path,
    assets: list[str] | None = None,
) -> pd.DataFrame:
    """
    Load daily OHLCV data from CSV or Parquet.

    Returns a DataFrame with a pd.MultiIndex on columns: (asset, field).
    Fields: open, high, low, close, volume (lowercase).
    Index: UTC DatetimeIndex.

    Validates:
    - No duplicate timestamps
    - No all-NaN rows
    - Monotonically increasing index

    For a single-asset CSV, expects columns: open, high, low, close, volume (+ optional date/timestamp column).
    For a multi-asset CSV, expects columns prefixed by asset name: BTC_close, ETH_close, etc.
    """
    assets = assets or ["BTC", "ETH"]
    path = Path(path)
    suffix = path.suffix.lower()

    if suffix == ".parquet":
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path, index_col=0, parse_dates=True)

    # Ensure UTC DatetimeIndex
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index, utc=True)
    elif df.index.tz is None:
        df.index = df.index.tz_localize("UTC")

    # Validate BEFORE any reordering
    if df.index.duplicated().any():
        raise ValueError("Duplicate timestamps found in data.")
    if not df.index.is_monotonic_increasing:
        raise ValueError("Index is not monotonically increasing.")
    if df.isnull().all(axis=1).any():
        raise ValueError("Data contains all-NaN rows.")

    # Build MultiIndex columns
    # Detect format: if columns already MultiIndex, use as-is
    if isinstance(df.columns, pd.MultiIndex):
        return df

    # Try to detect multi-asset format: BTC_close, ETH_open, etc.
    fields = ["open", "high", "low", "close", "volume"]
    tuples = []
    for col in df.columns:
        col_lower = col.lower()
        matched = False
        for asset in assets:
            for field in fields:
                if col_lower == f"{asset.lower()}_{field}" or col_lower == f"{asset.lower()}.{field}":
                    tuples.append((asset, field))
                    matched = True
                    break
            if matched:
                break
        if not matched:
            # Single-asset format: columns are just field names
            tuples = None
            break

    if tuples is not None and len(tuples) == len(df.columns):
        df.columns = pd.MultiIndex.from_tuples(tuples)
    else:
        # Single-asset format: wrap in MultiIndex with first asset
        asset = assets[0] if len(assets) >= 1 else "BTC"
        # Lowercase column names
        df.columns = [c.lower() for c in df.columns]
        df.columns = pd.MultiIndex.from_tuples([(asset, c) for c in df.columns])

    return df
