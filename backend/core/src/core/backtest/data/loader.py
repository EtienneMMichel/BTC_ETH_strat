from pathlib import Path
import pandas as pd


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
