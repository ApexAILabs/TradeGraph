"""Async HTTP clients for Finnhub (equities) and Binance (crypto)."""

from __future__ import annotations

import asyncio
from datetime import datetime
from typing import Any, Dict, List, Optional

import aiohttp
from loguru import logger


class FinnhubClient:
    BASE_URL = "https://finnhub.io/api/v1"

    def __init__(self, api_key: str, *, timeout: int = 20) -> None:
        if not api_key:
            raise ValueError("FINNHUB_API_KEY is required for market data")
        self.api_key = api_key
        self.timeout = timeout
        self._session: Optional[aiohttp.ClientSession] = None
        self._session_lock = asyncio.Lock()

    async def close(self) -> None:
        if self._session and not self._session.closed:
            await self._session.close()

    async def get_quote(self, symbol: str) -> Optional[Dict[str, Any]]:
        try:
            return await self._get_json("/quote", params={"symbol": symbol})
        except Exception as exc:
            logger.warning(f"Finnhub quote failed for {symbol}: {exc}")
            return None

    async def get_company_profile(self, symbol: str) -> Optional[Dict[str, Any]]:
        try:
            return await self._get_json("/stock/profile2", params={"symbol": symbol})
        except Exception as exc:
            logger.warning(f"Finnhub profile failed for {symbol}: {exc}")
            return None

    async def get_candles(
        self,
        symbol: str,
        *,
        resolution: str,
        start: datetime,
        end: datetime,
    ) -> Dict[str, Any]:
        params = {
            "symbol": symbol,
            "resolution": resolution,
            "from": int(start.timestamp()),
            "to": int(end.timestamp()),
        }
        return await self._get_json("/stock/candle", params=params)

    async def _get_json(
        self, path: str, params: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        session = await self._get_session()
        params = params.copy() if params else {}
        params["token"] = self.api_key
        url = f"{self.BASE_URL}{path}"
        async with session.get(url, params=params) as response:
            if response.status != 200:
                text = await response.text()
                raise RuntimeError(
                    f"Finnhub request failed ({response.status}) for {path}: {text[:200]}"
                )
            return await response.json()

    async def _get_session(self) -> aiohttp.ClientSession:
        async with self._session_lock:
            if self._session is None or self._session.closed:
                timeout = aiohttp.ClientTimeout(total=self.timeout)
                self._session = aiohttp.ClientSession(timeout=timeout)
            return self._session


class BinanceClient:
    BASE_URL = "https://api.binance.com"

    def __init__(self, *, timeout: int = 20) -> None:
        self.timeout = timeout
        self._session: Optional[aiohttp.ClientSession] = None
        self._session_lock = asyncio.Lock()

    async def close(self) -> None:
        if self._session and not self._session.closed:
            await self._session.close()

    async def get_price(self, symbol: str) -> Optional[float]:
        formatted = self._format_symbol(symbol)
        try:
            data = await self._get_json(
                "/api/v3/ticker/price", params={"symbol": formatted}
            )
            price = data.get("price") if isinstance(data, dict) else None
            return float(price) if price is not None else None
        except Exception as exc:
            logger.warning(f"Binance price failed for {symbol}: {exc}")
            return None

    async def get_klines(
        self,
        symbol: str,
        *,
        interval: str,
        limit: int = 500,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
    ) -> List[List[Any]]:
        formatted = self._format_symbol(symbol)
        params: Dict[str, Any] = {
            "symbol": formatted,
            "interval": interval,
            "limit": min(limit, 1000),
        }
        if start:
            params["startTime"] = int(start.timestamp() * 1000)
        if end:
            params["endTime"] = int(end.timestamp() * 1000)
        data = await self._get_json("/api/v3/klines", params=params)
        return data if isinstance(data, list) else []

    async def _get_json(
        self, path: str, params: Optional[Dict[str, Any]] = None
    ) -> Any:
        session = await self._get_session()
        url = f"{self.BASE_URL}{path}"
        async with session.get(url, params=params) as response:
            if response.status != 200:
                text = await response.text()
                raise RuntimeError(
                    f"Binance request failed ({response.status}) for {path}: {text[:200]}"
                )
            return await response.json()

    async def _get_session(self) -> aiohttp.ClientSession:
        async with self._session_lock:
            if self._session is None or self._session.closed:
                timeout = aiohttp.ClientTimeout(total=self.timeout)
                self._session = aiohttp.ClientSession(timeout=timeout)
            return self._session

    def _format_symbol(self, symbol: str) -> str:
        normalized = symbol.strip().upper().replace(":", "")
        if "-" in normalized:
            base, quote = normalized.split("-", 1)
        elif normalized.endswith("USDT"):
            return normalized
        elif normalized.endswith("USD"):
            base, quote = normalized[:-3], "USD"
        else:
            base, quote = normalized, "USDT"
        quote = "USDT" if quote in {"USD", "USDT"} else quote
        return f"{base}{quote}"
