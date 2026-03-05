from __future__ import annotations

"""Download candles and funding rate data from data.binance.vision."""
from pathlib import Path

SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT"]
YEARS   = [2022, 2023, 2024,2025,2026]
MONTHS  = range(1, 13)
RAW     = Path("../data/raw")

total = len(SYMBOLS) * len(YEARS) * 12
done  = 0
errors = []

from pathlib import Path


from .data.downloader import download_candles, download_funding


for sym in SYMBOLS:
    sym_dir = RAW / sym
    for yr in YEARS:
        for mo in MONTHS:
            done += 1
            tag = f"{sym} {yr}-{mo:02d}"

            candle_path = sym_dir / f"{sym}_1m_{yr}_{mo:02d}.parquet"
            if candle_path.exists():
                print(f"[{done:03d}/{total}] candles  SKIP {tag}", flush=True)
            else:
                try:
                    download_candles(sym, yr, mo, sym_dir)
                    print(f"[{done:03d}/{total}] candles  OK   {tag}", flush=True)
                except Exception as e:
                    errors.append(f"candles  {tag}: {e}")
                    print(f"[{done:03d}/{total}] candles  ERR  {tag}: {e}", flush=True)

            funding_path = sym_dir / f"{sym}_funding_{yr}_{mo:02d}.parquet"
            if funding_path.exists():
                print(f"[{done:03d}/{total}] funding  SKIP {tag}", flush=True)
            else:
                try:
                    download_funding(sym, yr, mo, sym_dir)
                    print(f"[{done:03d}/{total}] funding  OK   {tag}", flush=True)
                except Exception as e:
                    errors.append(f"funding  {tag}: {e}")
                    print(f"[{done:03d}/{total}] funding  ERR  {tag}: {e}", flush=True)

print(f"\nDone. {len(errors)} error(s).")
for e in errors:
    print(" ", e)
