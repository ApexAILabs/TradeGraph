"""Price trend utilities shared by websocket streams and PDF reports."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Optional, Sequence

from loguru import logger

from ..config.settings import settings
from .market_data_clients import FinnhubClient, BinanceClient


@dataclass(frozen=True)
class AggregateWindow:
    equity_resolution: str
    crypto_interval: str
    window_seconds: int
    limit: int = 500


class PriceTrendService:
    """Fetches historical pricing windows and builds normalized trend payloads."""

    def __init__(
        self,
        *,
        max_concurrent: int = 4,
        finnhub_client: Optional[FinnhubClient] = None,
        binance_client: Optional[BinanceClient] = None,
    ) -> None:
        self._semaphore = asyncio.Semaphore(max_concurrent)
        self._finnhub = finnhub_client or FinnhubClient(settings.finnhub_api_key)
        self._binance = binance_client or BinanceClient()
        self._owns_finnhub = finnhub_client is None
        self._owns_binance = binance_client is None
        self._timeframes: Dict[str, AggregateWindow] = {
            "last_month": AggregateWindow("60", "1h", 30 * 24 * 3600, limit=720),
            "last_week": AggregateWindow("30", "30m", 7 * 24 * 3600, limit=336),
            "last_day": AggregateWindow("5", "5m", 24 * 3600, limit=288),
            "last_hour": AggregateWindow("1", "1m", 3 * 3600, limit=180),
        }

    async def close(self) -> None:
        if self._owns_finnhub:
            await self._finnhub.close()
        if self._owns_binance:
            await self._binance.close()

    async def get_trends_for_symbols(
        self, symbols: Sequence[str]
    ) -> Dict[str, Dict[str, Any]]:
        normalized = [sym.strip().upper() for sym in symbols if sym]
        tasks = [self._run_symbol(symbol) for symbol in normalized]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        payload: Dict[str, Dict[str, Any]] = {}
        for symbol, result in zip(normalized, results):
            if isinstance(result, Exception):
                logger.warning(f"Trend calculation failed for {symbol}: {result}")
                continue
            if result:
                payload[symbol] = result
        return payload

    async def _run_symbol(self, symbol: str) -> Optional[Dict[str, Any]]:
        async with self._semaphore:
            now = datetime.now(timezone.utc)
            trend_tasks = {
                label: asyncio.create_task(self._fetch_trend(symbol, spec, now))
                for label, spec in self._timeframes.items()
            }
            results = await asyncio.gather(
                *trend_tasks.values(), return_exceptions=True
            )

            trends: Dict[str, Any] = {}
            for label, result in zip(trend_tasks.keys(), results):
                if isinstance(result, Exception):
                    logger.warning(
                        f"Trend window {label} failed for {symbol}: {result}"
                    )
                    continue
                if result:
                    trends[label] = result

            if not trends:
                return None

            snapshot: Dict[str, Any] = {
                "symbol": symbol,
                "trends": trends,
                "pricing_timestamp": now.isoformat(),
            }
            snapshot["current_price"] = self._determine_current_price(trends)
            return snapshot

    async def _fetch_trend(
        self, symbol: str, spec: AggregateWindow, now: datetime
    ) -> Optional[Dict[str, Any]]:
        if self._is_crypto(symbol):
            return await self._fetch_crypto_trend(symbol, spec, now)
        return await self._fetch_equity_trend(symbol, spec, now)

    async def _fetch_equity_trend(
        self, symbol: str, spec: AggregateWindow, now: datetime
    ) -> Optional[Dict[str, Any]]:
        start = now - timedelta(seconds=spec.window_seconds)
        try:
            candles = await self._finnhub.get_candles(
                symbol,
                resolution=spec.equity_resolution,
                start=start,
                end=now,
            )
        except Exception as exc:
            logger.warning(f"Finnhub aggregates failed for {symbol}: {exc}")
            return None
        if candles.get("s") != "ok":
            return None
        closes = candles.get("c", [])
        return self._summarize_series(closes)

    async def _fetch_crypto_trend(
        self, symbol: str, spec: AggregateWindow, now: datetime
    ) -> Optional[Dict[str, Any]]:
        start = now - timedelta(seconds=spec.window_seconds)
        try:
            klines = await self._binance.get_klines(
                symbol,
                interval=spec.crypto_interval,
                limit=spec.limit,
                start=start,
                end=now,
            )
        except Exception as exc:
            logger.warning(f"Binance klines failed for {symbol}: {exc}")
            return None
        if not klines:
            return None
        closes = [float(item[4]) for item in klines]
        return self._summarize_series(closes)

    def _determine_current_price(self, trends: Dict[str, Any]) -> Optional[float]:
        for key in ("last_hour", "last_day", "last_week", "last_month"):
            trend = trends.get(key)
            if trend and trend.get("end") is not None:
                return trend["end"]
        return None

    @staticmethod
    def _summarize_series(series: Sequence[float]) -> Optional[Dict[str, Any]]:
        values = [float(value) for value in series if value is not None]
        if len(values) < 2:
            return None
        start = values[0]
        end = values[-1]
        if start == 0:
            return None
        change = end - start
        pct_change = (change / start) * 100
        direction = "bullish" if change > 0 else ("bearish" if change < 0 else "flat")
        return {
            "start": start,
            "end": end,
            "change": change,
            "percent_change": pct_change,
            "direction": direction,
        }

    @staticmethod
    def _is_crypto(symbol: str) -> bool:
        normalized = symbol.upper()
        if normalized.startswith("X:") or normalized.startswith("CRYPTO:"):
            return True
        if "-" in normalized:
            _, suffix = normalized.split("-", 1)
            return suffix in {"USD", "USDT", "BTC", "ETH"}
        return False


__all__ = ["PriceTrendService"]
