"""Async client for Alpha Vantage market data APIs."""

from __future__ import annotations

import asyncio
from typing import Any, Dict, Optional

import aiohttp
from loguru import logger


class AlphaVantageClient:
    """Lightweight wrapper around the Alpha Vantage REST API."""

    BASE_URL = "https://www.alphavantage.co/query"

    def __init__(self, api_key: str, *, timeout: int = 20) -> None:
        if not api_key:
            raise ValueError("ALPHA_VANTAGE_API key is required to use Alpha Vantage")
        self.api_key = api_key
        self.timeout = timeout
        self._session: Optional[aiohttp.ClientSession] = None
        self._session_lock = asyncio.Lock()

    async def close(self) -> None:
        if self._session and not self._session.closed:
            await self._session.close()

    async def get_daily_time_series(self, symbol: str) -> Optional[Dict[str, Any]]:
        payload = await self._request(
            {
                "function": "TIME_SERIES_DAILY_ADJUSTED",
                "symbol": symbol,
                "outputsize": "compact",
            }
        )
        return self._extract_latest_bar(
            payload, series_key="Time Series (Daily)", include_adjusted=True
        )

    async def get_intraday_time_series(
        self, symbol: str, *, interval: str = "15min"
    ) -> Optional[Dict[str, Any]]:
        payload = await self._request(
            {
                "function": "TIME_SERIES_INTRADAY",
                "symbol": symbol,
                "interval": interval,
                "outputsize": "compact",
            }
        )
        series_key = f"Time Series ({interval})"
        return self._extract_latest_bar(payload, series_key)

    async def get_technical_indicator(
        self,
        symbol: str,
        *,
        indicator: str,
        interval: str = "daily",
        time_period: int = 20,
        series_type: str = "close",
    ) -> Optional[Dict[str, Any]]:
        indicator = indicator.upper()
        payload = await self._request(
            {
                "function": indicator,
                "symbol": symbol,
                "interval": interval,
                "time_period": time_period,
                "series_type": series_type,
            }
        )
        key = f"Technical Analysis: {indicator}"
        if not payload or key not in payload:
            return None
        values = payload[key]
        if not isinstance(values, dict):
            return None
        latest_timestamp = max(values.keys())
        latest_entry = values.get(latest_timestamp, {})
        parsed_values = {
            name: self._to_float(value) for name, value in latest_entry.items()
        }
        return {
            "indicator": indicator,
            "timestamp": latest_timestamp,
            "values": {
                key: value for key, value in parsed_values.items() if value is not None
            },
        }

    async def get_fx_rate(
        self, from_symbol: str, to_symbol: str
    ) -> Optional[Dict[str, Any]]:
        payload = await self._request(
            {
                "function": "CURRENCY_EXCHANGE_RATE",
                "from_currency": from_symbol,
                "to_currency": to_symbol,
            }
        )
        rate = payload.get("Realtime Currency Exchange Rate") if payload else None
        if not isinstance(rate, dict):
            return None
        return {
            "from_symbol": rate.get("1. From_Currency Code"),
            "to_symbol": rate.get("3. To_Currency Code"),
            "exchange_rate": self._to_float(rate.get("5. Exchange Rate")),
            "last_refreshed": rate.get("6. Last Refreshed"),
            "bid_price": self._to_float(rate.get("8. Bid Price")),
            "ask_price": self._to_float(rate.get("9. Ask Price")),
        }

    async def get_crypto_rate(
        self, symbol: str, market: str
    ) -> Optional[Dict[str, Any]]:
        # Alpha Vantage uses the same CURRENCY_EXCHANGE_RATE endpoint for FX and crypto
        return await self.get_fx_rate(symbol, market)

    async def get_sector_performance(self) -> Optional[Dict[str, Any]]:
        payload = await self._request({"function": "SECTOR"})
        return payload if isinstance(payload, dict) else None

    async def get_company_overview(self, symbol: str) -> Optional[Dict[str, Any]]:
        payload = await self._request({"function": "OVERVIEW", "symbol": symbol})
        if not isinstance(payload, dict) or not payload:
            return None
        return {
            "symbol": payload.get("Symbol", symbol),
            "name": payload.get("Name"),
            "description": payload.get("Description"),
            "sector": payload.get("Sector"),
            "industry": payload.get("Industry"),
            "market_cap": self._to_float(payload.get("MarketCapitalization")),
            "pe_ratio": self._to_float(payload.get("PERatio")),
            "eps": self._to_float(payload.get("EPS")),
            "revenue": self._to_float(payload.get("RevenueTTM")),
            "dividend_yield": self._to_float(payload.get("DividendYield")),
            "return_on_equity": self._to_float(payload.get("ReturnOnEquityTTM")),
            "return_on_assets": self._to_float(payload.get("ReturnOnAssetsTTM")),
            "debt_to_equity": self._to_float(payload.get("DebtToEquityRatio")),
            "current_ratio": self._to_float(payload.get("CurrentRatio")),
            "book_value": self._to_float(payload.get("BookValue")),
            "price_to_book": self._to_float(payload.get("PriceToBookRatio")),
            "beta": self._to_float(payload.get("Beta")),
            "fifty_two_week_high": self._to_float(payload.get("52WeekHigh")),
            "fifty_two_week_low": self._to_float(payload.get("52WeekLow")),
            "latest_quarter": payload.get("LatestQuarter"),
        }

    async def _request(self, params: Dict[str, Any]) -> Dict[str, Any]:
        session = await self._get_session()
        query = params.copy()
        query["apikey"] = self.api_key
        async with session.get(self.BASE_URL, params=query) as response:
            text = await response.text()
            if response.status != 200:
                raise RuntimeError(
                    f"Alpha Vantage request failed ({response.status}): {text[:200]}"
                )
            try:
                data = await response.json()
            except aiohttp.ContentTypeError:
                logger.warning(
                    f"Alpha Vantage returned a non-JSON response for {params.get('function')}"
                )
                return {}
        if isinstance(data, dict):
            if "Note" in data:
                logger.warning(
                    f"Alpha Vantage throttled request for {params.get('function')}: {data['Note']}"
                )
            if "Error Message" in data:
                logger.warning(
                    f"Alpha Vantage error for {params.get('function')}: {data['Error Message']}"
                )
        return data

    async def _get_session(self) -> aiohttp.ClientSession:
        async with self._session_lock:
            if self._session is None or self._session.closed:
                timeout = aiohttp.ClientTimeout(total=self.timeout)
                self._session = aiohttp.ClientSession(timeout=timeout)
            return self._session

    def _extract_latest_bar(
        self,
        payload: Optional[Dict[str, Any]],
        series_key: str,
        include_adjusted: bool = False,
    ) -> Optional[Dict[str, Any]]:
        if not payload:
            return None
        series = payload.get(series_key)
        if not isinstance(series, dict) or not series:
            return None
        latest_timestamp = max(series.keys())
        entry = series.get(latest_timestamp, {})
        bar = {
            "timestamp": latest_timestamp,
            "open": self._to_float(entry.get("1. open")),
            "high": self._to_float(entry.get("2. high")),
            "low": self._to_float(entry.get("3. low")),
            "close": self._to_float(entry.get("4. close")),
            "volume": self._to_float(entry.get("5. volume")),
        }
        if include_adjusted:
            bar["adjusted_close"] = self._to_float(
                entry.get("5. adjusted close") or entry.get("6. adjusted close")
            )
        return bar

    @staticmethod
    def _to_float(value: Any) -> Optional[float]:
        try:
            if value is None or value == "":
                return None
            return float(value)
        except (ValueError, TypeError):  # pragma: no cover - defensive
            return None
