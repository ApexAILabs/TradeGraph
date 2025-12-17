"""Streaming channel infrastructure for multi-source financial news and pricing."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Sequence

import aiohttp
import feedparser
from loguru import logger

from ..utils.helpers import generate_summary
from .price_trend_service import PriceTrendService


DEFAULT_HEADERS = {
    "Accept": "application/rss+xml,application/xml;q=0.9,*/*;q=0.8",
    "User-Agent": "RADGEGRAPHBot/1.0 (https://github.com/Mehranmzn/RADGEGRAPH)",
}


@dataclass
class NewsSource:
    """Metadata describing a remote news source."""

    id: str
    name: str
    url: str
    coverage: str
    topics: List[str]
    category: str
    is_open_access: bool = True
    supports_crypto: bool = False


@dataclass
class ChannelDefinition:
    """Configuration for a websocket enabled channel."""

    channel_id: str
    title: str
    description: str
    refresh_seconds: int
    sources: List[NewsSource] = field(default_factory=list)
    stream_type: str = "news"
    default_symbols: Sequence[str] = field(default_factory=lambda: ("AAPL", "MSFT"))

    def metadata(self) -> Dict[str, Any]:
        return {
            "channel_id": self.channel_id,
            "title": self.title,
            "description": self.description,
            "refresh_seconds": self.refresh_seconds,
            "stream_type": self.stream_type,
            "default_symbols": list(self.default_symbols),
            "source_count": len(self.sources),
            "sources": [asdict(source) for source in self.sources],
        }


class ChannelType(str, Enum):
    """Supported channel identifiers."""

    TOP_MARKET_CRYPTO = "top_market_crypto"
    OPEN_SOURCE_AGENCIES = "open_source_agencies"
    LIVE_PRICE_STREAM = "live_price_stream"

    @classmethod
    def from_value(cls, value: str) -> "ChannelType":
        for member in cls:
            if member.value == value:
                return member
        raise ValueError(f"Unsupported channel: {value}")


CHANNEL_REGISTRY: Dict[ChannelType, ChannelDefinition] = {
    ChannelType.TOP_MARKET_CRYPTO: ChannelDefinition(
        channel_id=ChannelType.TOP_MARKET_CRYPTO.value,
        title="Top Market & Crypto Headlines",
        description=(
            "Aggregates market-moving stories from Reuters, CNBC, Wall Street Journal, "
            "MarketWatch, and CoinDesk to cover both equities and crypto."
        ),
        refresh_seconds=45,
        sources=[
            NewsSource(
                id="reuters",
                name="Reuters Top News",
                url="https://feeds.reuters.com/reuters/topNews",
                coverage="Global business and markets",
                topics=["stocks", "macro", "economy"],
                category="tier_one",
            ),
            NewsSource(
                id="cnbc",
                name="CNBC Markets",
                url="https://www.cnbc.com/id/100003114/device/rss/rss.html",
                coverage="US and global equity markets",
                topics=["stocks", "earnings", "macro"],
                category="tier_one",
            ),
            NewsSource(
                id="wsj",
                name="WSJ Markets",
                url="https://feeds.a.dj.com/rss/RSSMarketsMain.xml",
                coverage="Wall Street Journal markets desk",
                topics=["stocks", "policy", "macro"],
                category="tier_one",
            ),
            NewsSource(
                id="marketwatch",
                name="MarketWatch Top Stories",
                url="https://www.marketwatch.com/rss/topstories",
                coverage="MarketWatch newsroom",
                topics=["stocks", "factors", "macro"],
                category="tier_one",
            ),
            NewsSource(
                id="coindesk",
                name="CoinDesk",
                url="https://www.coindesk.com/arc/outboundfeeds/rss/",
                coverage="Digital assets and blockchain",
                topics=["crypto", "regulation"],
                category="crypto",
                supports_crypto=True,
            ),
        ],
    ),
    ChannelType.OPEN_SOURCE_AGENCIES: ChannelDefinition(
        channel_id=ChannelType.OPEN_SOURCE_AGENCIES.value,
        title="Open News Agencies",
        description=(
            "Five open-license newsrooms (The Guardian, BBC Business, Al Jazeera, NPR, "
            "and Financial Express) for freely accessible economic reporting."
        ),
        refresh_seconds=60,
        sources=[
            NewsSource(
                id="guardian",
                name="The Guardian Business",
                url="https://www.theguardian.com/business/rss",
                coverage="Guardian Open Platform (CC BY)",
                topics=["global", "policy", "companies"],
                category="open_agency",
            ),
            NewsSource(
                id="bbc",
                name="BBC Business",
                url="https://feeds.bbci.co.uk/news/business/rss.xml",
                coverage="BBC World Service",
                topics=["economy", "markets"],
                category="open_agency",
            ),
            NewsSource(
                id="aljazeera",
                name="Al Jazeera Economy",
                url="https://www.aljazeera.com/xml/rss/all.xml",
                coverage="Global south lens",
                topics=["emerging", "energy"],
                category="open_agency",
            ),
            NewsSource(
                id="npr",
                name="NPR Economy",
                url="https://www.npr.org/rss/rss.php?id=1006",
                coverage="US economy (Creative Commons)",
                topics=["policy", "inflation"],
                category="open_agency",
            ),
            NewsSource(
                id="financialexpress",
                name="Financial Express Markets",
                url="https://www.financialexpress.com/feed/market/",
                coverage="Free-to-read Indian markets coverage",
                topics=["asia", "markets"],
                category="open_agency",
            ),
        ],
    ),
    ChannelType.LIVE_PRICE_STREAM: ChannelDefinition(
        channel_id=ChannelType.LIVE_PRICE_STREAM.value,
        title="Live Price & Trend Stream",
        description=(
            "Combines Finnhub equity aggregates and Binance crypto klines with RADGEGRAPH trend analytics "
            "to serve last year/month/day/hour performance for equities and crypto."
        ),
        refresh_seconds=30,
        stream_type="prices",
        default_symbols=("AAPL", "MSFT", "BTC-USD", "ETH-USD"),
    ),
}


class FinancialNewsChannelService:
    """Fetches and structures channel payloads for websocket consumers."""

    def __init__(
        self,
        *,
        max_items_per_source: int = 5,
        max_items_per_channel: int = 25,
        price_service: Optional[PriceTrendService] = None,
        include_open_agencies: bool = False,
    ) -> None:
        self.max_items_per_source = max_items_per_source
        self.max_items_per_channel = max_items_per_channel
        self.price_service = price_service or PriceTrendService()
        self.include_open_agencies = include_open_agencies
        self._session_lock = asyncio.Lock()
        self._session: Optional[aiohttp.ClientSession] = None

    async def close(self) -> None:
        if self._session and not self._session.closed:
            await self._session.close()

    async def _get_session(self) -> aiohttp.ClientSession:
        async with self._session_lock:
            if self._session is None or self._session.closed:
                timeout = aiohttp.ClientTimeout(total=20)
                self._session = aiohttp.ClientSession(timeout=timeout)
            return self._session

    def describe_channels(self) -> List[Dict[str, Any]]:
        return [
            CHANNEL_REGISTRY[channel].metadata() for channel in self._iter_channels()
        ]

    async def collect_all_channels(
        self, symbols: Optional[Sequence[str]] = None
    ) -> Dict[str, Any]:
        tasks = [
            self.fetch_channel_payload(channel.value, symbols)
            for channel in self._iter_channels()
        ]
        payloads = await asyncio.gather(*tasks, return_exceptions=True)

        result: Dict[str, Any] = {}
        for channel, payload in zip(self._iter_channels(), payloads):
            if isinstance(payload, Exception):
                logger.warning(f"Failed to collect channel {channel.value}: {payload}")
                continue
            result[channel.value] = payload
        return result

    def _iter_channels(self):
        for channel in ChannelType:
            if (
                not self.include_open_agencies
                and channel == ChannelType.OPEN_SOURCE_AGENCIES
            ):
                continue
            yield channel

    async def fetch_channel_payload(
        self, channel_id: str, symbols: Optional[Sequence[str]] = None
    ) -> Dict[str, Any]:
        channel_type = ChannelType.from_value(channel_id)
        channel_def = CHANNEL_REGISTRY[channel_type]
        normalized_symbols = self._normalize_symbols(symbols) or [
            sym.upper() for sym in channel_def.default_symbols
        ]

        if channel_def.stream_type == "prices":
            items = await self._build_price_items(normalized_symbols)
        else:
            items = await self._collect_news(channel_def, normalized_symbols)

        return {
            "channel_id": channel_def.channel_id,
            "title": channel_def.title,
            "description": channel_def.description,
            "fetched_at": datetime.now(timezone.utc).isoformat(),
            "symbols": normalized_symbols,
            "items": items,
            "source_count": len(channel_def.sources),
        }

    async def _collect_news(
        self, channel_definition: ChannelDefinition, symbols: Sequence[str]
    ) -> List[Dict[str, Any]]:
        session = await self._get_session()
        tasks = [
            self._fetch_source_news(session, source, symbols)
            for source in channel_definition.sources
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        collected: List[Dict[str, Any]] = []
        for source, payload in zip(channel_definition.sources, results):
            if isinstance(payload, Exception):
                logger.warning(f"Failed to fetch feed from {source.name}: {payload}")
                continue
            collected.extend(payload)

        collected.sort(
            key=lambda item: item.get("published_at", ""),
            reverse=True,
        )
        return collected[: self.max_items_per_channel]

    async def _fetch_source_news(
        self,
        session: aiohttp.ClientSession,
        source: NewsSource,
        symbols: Sequence[str],
    ) -> List[Dict[str, Any]]:
        try:
            async with session.get(source.url, headers=DEFAULT_HEADERS) as response:
                if response.status == 403:
                    logger.info(
                        f"{source.name} feed blocked with 403. Skipping until access is restored."
                    )
                    return []
                if response.status != 200:
                    logger.warning(
                        f"{source.name} feed returned status {response.status}. Skipping."
                    )
                    return []
                feed_body = await response.text()
        except aiohttp.ClientConnectorError as exc:
            logger.info(f"{source.name} feed unreachable: {exc}")
            return []
        except Exception as exc:
            logger.warning(f"{source.name} feed failed: {exc}")
            return []

        feed = await asyncio.to_thread(feedparser.parse, feed_body)
        articles: List[Dict[str, Any]] = []

        for entry in feed.entries[: self.max_items_per_source * 2]:
            normalized = self._normalize_entry(entry, source, symbols)
            if not normalized:
                continue
            articles.append(normalized)

        return articles[: self.max_items_per_source]

    def _normalize_entry(
        self, entry: Any, source: NewsSource, symbols: Sequence[str]
    ) -> Optional[Dict[str, Any]]:
        title = entry.get("title") or ""
        summary = entry.get("summary") or entry.get("description") or ""
        link = entry.get("link") or source.url

        matched_symbols = [
            symbol
            for symbol in symbols
            if symbol in title.upper() or symbol in summary.upper()
        ]

        if symbols and not matched_symbols and source.category != "open_agency":
            # For top-tier feeds, keep only relevant tickers when provided.
            return None

        published = None
        published_data = entry.get("published_parsed") or entry.get("updated_parsed")
        if published_data:
            published = datetime(*published_data[:6], tzinfo=timezone.utc)

        summarized = generate_summary(summary)
        tags = [
            tag.get("term")
            for tag in entry.get("tags", [])
            if isinstance(tag, dict) and tag.get("term")
        ]

        return {
            "title": title.strip(),
            "summary": summarized,
            "raw_summary": summary.strip(),
            "url": link,
            "source": source.name,
            "source_id": source.id,
            "coverage": source.coverage,
            "topics": list({*source.topics, *(tags or [])}),
            "matched_symbols": matched_symbols,
            "published_at": published.isoformat() if published else None,
        }

    async def _build_price_items(self, symbols: Sequence[str]) -> List[Dict[str, Any]]:
        trends = await self.price_service.get_trends_for_symbols(list(symbols))
        items: List[Dict[str, Any]] = []
        for symbol in symbols:
            symbol_data = trends.get(symbol)
            if not symbol_data:
                continue
            items.append(
                {
                    "symbol": symbol,
                    "current_price": symbol_data.get("current_price"),
                    "pricing_timestamp": symbol_data.get("pricing_timestamp"),
                    "trends": symbol_data.get("trends", {}),
                }
            )
        return items

    @staticmethod
    def _normalize_symbols(symbols: Optional[Sequence[str]]) -> List[str]:
        if not symbols:
            return []
        return [sym.strip().upper() for sym in symbols if sym and sym.strip()]


__all__ = [
    "FinancialNewsChannelService",
    "ChannelType",
    "ChannelDefinition",
    "NewsSource",
    "CHANNEL_REGISTRY",
]
