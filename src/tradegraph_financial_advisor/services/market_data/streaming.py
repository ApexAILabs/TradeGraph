"""WebSocket-based market data feeds used across the project."""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, Optional

import aiohttp
from aiohttp import ClientSession, WSMsgType
from loguru import logger


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


@dataclass
class RealtimeTrade:
    """Represents the latest trade received from a WebSocket feed."""

    symbol: str
    price: float
    size: Optional[float]
    timestamp: datetime
    provider: str
    raw: Dict[str, Any] = field(default_factory=dict)

    @property
    def latency_ms(self) -> float:
        """Approximate feed latency in milliseconds."""
        return max((_utc_now() - self.timestamp).total_seconds() * 1000, 0.0)


@dataclass
class MarketDataFeedConfig:
    """Configuration for selecting WebSocket feeds."""

    equity_provider: str = "polygon"
    crypto_provider: str = "binance"
    polygon_api_key: Optional[str] = None
    alpaca_api_key: Optional[str] = None
    alpaca_api_secret: Optional[str] = None
    alpaca_feed: str = "iex"
    timeout_seconds: int = 8

    def provider_for(self, asset_type: str, override: Optional[str] = None) -> str:
        asset_type = asset_type.lower()
        if override:
            return override.lower()
        if asset_type == "crypto":
            return self.crypto_provider.lower()
        return self.equity_provider.lower()


class WebSocketFeedError(RuntimeError):
    """Raised when a WebSocket feed cannot be used."""


class MarketDataWebSocketClient:
    """Small helper that fetches single-trade snapshots from vendor feeds."""

    def __init__(
        self,
        session: Optional[ClientSession] = None,
        config: Optional[MarketDataFeedConfig] = None,
    ) -> None:
        self._session = session
        self._owns_session = session is None
        self.config = config or MarketDataFeedConfig()

    async def __aenter__(self) -> "MarketDataWebSocketClient":
        await self._ensure_session()
        return self

    async def __aexit__(self, *exc_info: Any) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        if self._owns_session and self._session:
            await self._session.close()
            self._session = None

    async def _ensure_session(self) -> ClientSession:
        if self._session is None or self._session.closed:
            timeout = aiohttp.ClientTimeout(total=self.config.timeout_seconds + 2)
            self._session = ClientSession(timeout=timeout)
            self._owns_session = True
        return self._session

    async def get_realtime_trade(
        self,
        symbol: str,
        asset_type: str = "equity",
        provider_override: Optional[str] = None,
    ) -> RealtimeTrade:
        """Fetch the next available trade for the requested symbol."""

        provider = self.config.provider_for(asset_type, provider_override)
        symbol = symbol.upper()

        if provider == "binance":
            return await self._fetch_from_binance(symbol)
        if provider == "polygon":
            return await self._fetch_from_polygon(symbol)
        if provider == "alpaca":
            return await self._fetch_from_alpaca(symbol)

        raise WebSocketFeedError(f"Unsupported WebSocket provider: {provider}")

    async def _fetch_from_binance(self, symbol: str) -> RealtimeTrade:
        session = await self._ensure_session()
        stream_symbol = self._format_binance_symbol(symbol)
        url = f"wss://stream.binance.com:9443/ws/{stream_symbol}@trade"
        logger.debug(f"Connecting to Binance WebSocket for {stream_symbol}")

        try:
            async with session.ws_connect(url, heartbeat=45) as ws:
                payload = await self._next_json(ws)
                trade_event = payload
                if "data" in payload:  # combined stream response
                    trade_event = payload["data"]

                event_time = datetime.fromtimestamp(
                    trade_event.get("E", trade_event.get("eventTime", 0)) / 1000,
                    tz=timezone.utc,
                )
                return RealtimeTrade(
                    symbol=symbol,
                    price=float(trade_event["p"]),
                    size=float(trade_event.get("q") or 0.0),
                    timestamp=event_time,
                    provider="binance",
                    raw=trade_event,
                )
        except asyncio.TimeoutError as exc:
            raise WebSocketFeedError("Timed out waiting for Binance trade data") from exc
        except aiohttp.ClientError as exc:
            raise WebSocketFeedError(f"Binance WebSocket error: {exc}") from exc

    def _format_binance_symbol(self, symbol: str) -> str:
        normalized = symbol.replace("/", "").lower()
        if not normalized.endswith("usdt"):
            normalized = f"{normalized}usdt"
        return normalized

    async def _fetch_from_polygon(self, symbol: str) -> RealtimeTrade:
        if not self.config.polygon_api_key:
            raise WebSocketFeedError(
                "POLYGON_API_KEY is required for Polygon WebSocket access"
            )

        session = await self._ensure_session()
        url = "wss://socket.polygon.io/stocks"
        logger.debug(f"Connecting to Polygon WebSocket for {symbol}")

        try:
            async with session.ws_connect(url, heartbeat=30) as ws:
                await ws.send_json({"action": "auth", "params": self.config.polygon_api_key})
                await ws.send_json({"action": "subscribe", "params": f"T.{symbol}"})

                while True:
                    payload = await self._next_json(ws)
                    events = payload if isinstance(payload, list) else [payload]
                    for event in events:
                        if event.get("ev") != "T":
                            continue
                        event_time = datetime.fromtimestamp(event["t"] / 1_000_000_000, tz=timezone.utc)
                        return RealtimeTrade(
                            symbol=symbol,
                            price=float(event["p"]),
                            size=float(event.get("s") or 0.0),
                            timestamp=event_time,
                            provider="polygon",
                            raw=event,
                        )
        except asyncio.TimeoutError as exc:
            raise WebSocketFeedError("Timed out waiting for Polygon trade data") from exc
        except aiohttp.ClientError as exc:
            raise WebSocketFeedError(f"Polygon WebSocket error: {exc}") from exc

    async def _fetch_from_alpaca(self, symbol: str) -> RealtimeTrade:
        if not self.config.alpaca_api_key or not self.config.alpaca_api_secret:
            raise WebSocketFeedError(
                "ALPACA_API_KEY and ALPACA_API_SECRET are required for Alpaca WebSocket access"
            )

        feed = self.config.alpaca_feed
        session = await self._ensure_session()
        url = f"wss://stream.data.alpaca.markets/v2/{feed}"
        logger.debug(f"Connecting to Alpaca {feed} feed for {symbol}")

        try:
            async with session.ws_connect(url, heartbeat=30) as ws:
                await ws.send_json(
                    {
                        "action": "auth",
                        "key": self.config.alpaca_api_key,
                        "secret": self.config.alpaca_api_secret,
                    }
                )
                await ws.send_json({"action": "subscribe", "trades": [symbol], "quotes": [], "bars": []})

                while True:
                    payload = await self._next_json(ws)
                    events = payload if isinstance(payload, list) else [payload]
                    for event in events:
                        if event.get("T") != "t":
                            continue
                        timestamp = event.get("t")
                        if isinstance(timestamp, str):
                            event_time = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
                        else:
                            event_time = datetime.fromtimestamp(timestamp, tz=timezone.utc)
                        return RealtimeTrade(
                            symbol=symbol,
                            price=float(event.get("p")),
                            size=float(event.get("s") or 0.0),
                            timestamp=event_time,
                            provider="alpaca",
                            raw=event,
                        )
        except asyncio.TimeoutError as exc:
            raise WebSocketFeedError("Timed out waiting for Alpaca trade data") from exc
        except aiohttp.ClientError as exc:
            raise WebSocketFeedError(f"Alpaca WebSocket error: {exc}") from exc

    async def _next_json(self, ws: aiohttp.ClientWebSocketResponse) -> Any:
        msg = await asyncio.wait_for(ws.receive(), timeout=self.config.timeout_seconds)
        if msg.type == WSMsgType.TEXT:
            return json.loads(msg.data)
        if msg.type == WSMsgType.BINARY:
            return json.loads(msg.data.decode("utf-8"))
        if msg.type in (WSMsgType.CLOSED, WSMsgType.ERROR):
            raise WebSocketFeedError("WebSocket connection closed unexpectedly")
        return {}
