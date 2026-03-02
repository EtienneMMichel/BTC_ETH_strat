# data — Historical Downloader, Loader & Live WebSocket

Responsible for getting market data into the system, both historically and in real time.

## Responsibilities
- Download 1-minute OHLCV candles + funding rates from `data.binance.vision` as Parquet
- Cache data locally under `data/raw/<symbol>/`
- Stream rows from Parquet files as `MarketEvent` objects for backtesting
- Connect to Binance WebSocket streams for live kline + funding rate data
- Maintain the universe of tradeable symbols

## Files to implement
| File | Purpose |
|------|---------|
| `downloader.py` | Downloads candle zips + funding CSVs, converts to Parquet, caches locally |
| `loader.py` | Reads Parquet via `polars.LazyFrame`, yields `MarketEvent` row by row |
| `websocket_client.py` | Async Binance WebSocket — kline + funding streams → `MarketEvent` |
| `universe.py` | Top-30 USDT perp symbols (hardcoded list + REST refresh) |

## Storage layout
```
data/
└── raw/
    └── BTCUSDT/
        ├── candles_1m_2023.parquet
        └── funding_rate_2023.parquet
```

## Dependencies
- `polars`, `pyarrow` — Parquet I/O
- `requests`, `httpx` — HTTP downloads
- `websockets` — live streaming
- `core.events` — emits `MarketEvent`

## Test file
`tests/test_data.py`
