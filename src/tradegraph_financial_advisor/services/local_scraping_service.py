"""Local scraping helpers that supplement API-based news collection."""

from __future__ import annotations

import asyncio
from typing import Any, Dict, List, Sequence, Optional
from urllib.parse import urlparse

from loguru import logger
from ddgs import DDGS
from crawl4ai import AsyncWebCrawler

from ..models.financial_data import NewsArticle
from ..utils.helpers import generate_summary
from ..repositories import NewsRepository


class LocalScrapingService:
    OPEN_AGENCY_DOMAINS = {
        "theguardian.com",
        "guardian.co.uk",
        "bbc.co.uk",
        "bbci.co.uk",
        "aljazeera.com",
        "npr.org",
        "financialexpress.com",
    }

    def __init__(self, news_repository: Optional[NewsRepository] = None):
        self.crawler = AsyncWebCrawler()
        self.news_repository = news_repository or self._build_repository()

    async def search_and_scrape_news(
        self, symbols: Sequence[str], max_articles_per_symbol: int = 5
    ) -> List[NewsArticle]:
        """Use ddgs + Crawl4AI to pull supplemental ticker news."""

        all_articles: List[NewsArticle] = []
        if not symbols:
            return all_articles

        new_articles: List[NewsArticle] = []

        with DDGS() as ddgs:
            for symbol in symbols:
                query = f"{symbol} stock news"
                try:
                    results = await self._ddgs_news(
                        ddgs,
                        query=query,
                        region="us-en",
                        safesearch="off",
                        timelimit="d",
                        max_results=max_articles_per_symbol,
                    )
                except Exception as exc:
                    logger.error(f"Failed ddgs news search for {symbol}: {exc}")
                    continue

                for result in results:
                    try:
                        url = result.get("url") or result.get("href")
                        if not url:
                            continue
                        netloc = urlparse(url).netloc.lower()
                        if any(
                            netloc.endswith(domain)
                            for domain in self.OPEN_AGENCY_DOMAINS
                        ):
                            continue
                        scraped_data = await self.crawler.arun(url)
                        if not scraped_data or not scraped_data.markdown:
                            continue
                        article = NewsArticle(
                            title=result.get("title", ""),
                            url=url,
                            content=scraped_data.markdown,
                            summary=generate_summary(scraped_data.markdown),
                            source=result.get("source", "ddgs"),
                            published_at=result.get("date", ""),
                            symbols=[symbol],
                        )
                        all_articles.append(article)
                        new_articles.append(article)
                    except Exception as scrape_exc:
                        logger.warning(f"Failed to scrape article {result.get('url')}: {scrape_exc}")

        if new_articles:
            await self._persist_articles(new_articles)
        return all_articles

    async def search_and_scrape_financial_reports(
        self, company_symbol: str, report_type: str = "10-K"
    ) -> List[Dict[str, Any]]:
        query = f"{company_symbol} {report_type} site:sec.gov"
        filings: List[Dict[str, Any]] = []

        with DDGS() as ddgs:
            try:
                results = await self._ddgs_text(
                    ddgs,
                    query=query,
                    region="us-en",
                    safesearch="off",
                    max_results=5,
                )
            except Exception as exc:
                logger.error(
                    f"Failed ddgs filing search for {company_symbol} {report_type}: {exc}"
                )
                return filings

            for result in results:
                href = result.get("href") or result.get("url")
                if not href:
                    continue
                try:
                    scraped_data = await self.crawler.arun(href)
                    if scraped_data and scraped_data.markdown:
                        filings.append(
                            {
                                "url": href,
                                "content": scraped_data.markdown,
                                "report_type": report_type,
                            }
                        )
                except Exception as scrape_exc:
                    logger.warning(f"Failed to scrape report {href}: {scrape_exc}")

        return filings

    async def start(self):
        logger.info("LocalScrapingService started.")
        await self.crawler.start()

    async def stop(self):
        logger.info("LocalScrapingService stopped.")
        await self.crawler.close()
        self.news_repository = None

    async def health_check(self) -> bool:
        # For now, we assume the service is healthy if it can be instantiated.
        return True

    async def _ddgs_news(
        self, ddgs: DDGS, *, query: str, **kwargs: Any
    ) -> List[Dict[str, Any]]:
        """Run blocking DDGS.news in a worker thread."""

        def _runner() -> List[Dict[str, Any]]:
            return list(ddgs.news(query, **kwargs))

        return await asyncio.to_thread(_runner)

    async def _ddgs_text(
        self, ddgs: DDGS, *, query: str, **kwargs: Any
    ) -> List[Dict[str, Any]]:
        """Run blocking DDGS.text in a worker thread."""

        def _runner() -> List[Dict[str, Any]]:
            return list(ddgs.text(query, **kwargs))

        return await asyncio.to_thread(_runner)

    def _build_repository(self) -> Optional[NewsRepository]:
        try:
            return NewsRepository()
        except Exception as exc:
            logger.warning(f"News repository initialization failed: {exc}")
            return None

    async def _persist_articles(self, articles: List[NewsArticle]) -> None:
        if not self.news_repository or not articles:
            return
        try:
            await asyncio.to_thread(self.news_repository.record_articles, articles)
        except Exception as exc:
            logger.warning(f"Failed to write scraped news to DuckDB: {exc}")
