from __future__ import annotations
import asyncio
import json
from datetime import datetime, timezone
from typing import Callable, Optional

import websockets

from trading.core.events import MarketEvent

WS_BASE = "wss://fstream.binance.com/ws"


class BinanceWebSocketClient:
    def __init__(self, symbols: list[str], on_event: Callable[[MarketEvent], None]) -> None:
        self.symbols = symbols
        self.on_event = on_event
        self._running = False

    async def connect(self) -> None:
        streams = "/".join(f"{s.lower()}@kline_1m" for s in self.symbols)
        url = f"{WS_BASE}/{streams}"
        self._running = True
        async with websockets.connect(url) as ws:
            while self._running:
                try:
                    raw = await asyncio.wait_for(ws.recv(), timeout=30)
                    self._handle(raw)
                except asyncio.TimeoutError:
                    continue

    def stop(self) -> None:
        self._running = False

    def _handle(self, raw: str) -> None:
        msg = json.loads(raw)
        data = msg.get("data", msg)
        k = data.get("k", {})
        if not k.get("x"):  # only closed candles
            return
        event = MarketEvent(
            timestamp=datetime.fromtimestamp(k["t"] / 1000, tz=timezone.utc).replace(tzinfo=None),
            symbol=k["s"],
            open=float(k["o"]),
            high=float(k["h"]),
            low=float(k["l"]),
            close=float(k["c"]),
            volume=float(k["v"]),
        )
        self.on_event(event)
