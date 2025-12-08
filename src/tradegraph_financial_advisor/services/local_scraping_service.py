from datetime import datetime
from typing import Any, Dict, List, Optional

from loguru import logger
from ddgs import DDGS
from crawl4ai import AsyncWebCrawler

from ..models.financial_data import NewsArticle
from ..utils.helpers import generate_summary


class LocalScrapingService:
    """Bridges DuckDuckGo Search + Crawl4AI, with graceful fallbacks when Playwright is missing."""

    def __init__(self):
        self.crawler: Optional[AsyncWebCrawler] = None
        self._crawler_ready = False

    async def search_and_scrape_news(
        self, symbols: List[str], max_articles_per_symbol: int = 5
    ) -> List[NewsArticle]:
        all_articles = []
        ddgs = DDGS()
        for symbol in symbols:
            query = f"{symbol} stock news"
            try:
                results = ddgs.news(
                    query,
                    region="us-en",
                    safesearch="off",
                    timelimit="d",
                    max_results=max_articles_per_symbol,
                )
                if results:
                    for result in results:
                        try:
                            article = await self._build_article_from_result(symbol, result)
                            if article:
                                all_articles.append(article)
                        except Exception as e:
                            logger.warning(
                                f"Failed to process article {result.get('url') or result.get('href')}: {e}"
                            )
            except Exception as e:
                logger.error(f"Failed to search for news for symbol {symbol}: {e}")
        return all_articles

    async def search_and_scrape_financial_reports(
        self, company_symbol: str, report_type: str = "10-K"
    ) -> List[Dict[str, Any]]:
        query = f"{company_symbol} {report_type} site:sec.gov"
        filings = []
        ddgs = DDGS()
        try:
            results = ddgs.text(
                query,
                region="us-en",
                safesearch="off",
                max_results=5,
            )
            if results:
                for result in results:
                    try:
                        filing = await self._build_filing_from_result(result, report_type)
                        if filing:
                            filings.append(filing)
                    except Exception as e:
                        logger.warning(
                            f"Failed to scrape report {result.get('href')}: {e}"
                        )
        except Exception as e:
            logger.error(
                f"Failed to search for financial reports for symbol {company_symbol}: {e}"
            )
        return filings

    async def start(self):
        if self._crawler_ready:
            return
        try:
            if not self.crawler:
                self.crawler = AsyncWebCrawler()
            await self.crawler.start()
            self._crawler_ready = True
            logger.info("LocalScrapingService started.")
        except Exception as e:
            self.crawler = None
            self._crawler_ready = False
            logger.warning(
                "LocalScrapingService running without Playwright crawler: {}", e
            )

    async def stop(self):
        if self.crawler and self._crawler_ready:
            try:
                await self.crawler.close()
            except Exception:
                pass
        self._crawler_ready = False
        logger.info("LocalScrapingService stopped.")

    async def health_check(self) -> bool:
        # For now, we assume the service is healthy if it can be instantiated.
        return True

    async def _build_article_from_result(self, symbol: str, result: Dict[str, Any]) -> Optional[NewsArticle]:
        """Return a NewsArticle either by crawling the page or falling back to the snippet."""

        url = result.get("url") or result.get("href") or ""
        source = result.get("source", "ddgs")
        published_at = result.get("date") or datetime.now().isoformat()

        content = ""
        if self._crawler_ready and self.crawler and url:
            try:
                scraped = await self.crawler.arun(url)
                if scraped and scraped.markdown:
                    content = scraped.markdown
            except Exception as e:
                logger.warning(f"Crawler failed for {url}: {e}")

        if not content:
            content = result.get("body") or result.get("snippet") or result.get("title", "")

        summary = generate_summary(content) if content else result.get("title", "")

        return NewsArticle(
            title=result.get("title", url),
            url=url,
            content=content,
            summary=summary,
            source=source,
            published_at=published_at,
            symbols=[symbol],
        )

    async def _build_filing_from_result(self, result: Dict[str, Any], report_type: str) -> Optional[Dict[str, Any]]:
        url = result.get("href") or result.get("url")
        if not url:
            return None
        content = ""
        if self._crawler_ready and self.crawler:
            try:
                scraped = await self.crawler.arun(url)
                if scraped and scraped.markdown:
                    content = scraped.markdown
            except Exception as e:
                logger.warning(f"Crawler failed for filing {url}: {e}")

        if not content:
            content = result.get("body") or result.get("snippet") or ""

        return {
            "url": url,
            "content": content,
            "report_type": report_type,
        }
