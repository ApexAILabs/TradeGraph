from typing import Any, Dict, List
from loguru import logger
from ddgs import DDGS
from crawl4ai import AsyncWebCrawler
from ..models.financial_data import NewsArticle
from ..utils.helpers import generate_summary

class LocalScrapingService:
    def __init__(self):
        self.crawler = AsyncWebCrawler()

    async def search_and_scrape_news(
        self, symbols: List[str], max_articles_per_symbol: int = 5
    ) -> List[NewsArticle]:
        all_articles = []
        async with DDGS() as ddgs:
            for symbol in symbols:
                query = f"{symbol} stock news"
                try:
                    results = await ddgs.news(
                        keywords=query,
                        region="us-en",
                        safesearch="off",
                        timelimit="d",
                        max_results=max_articles_per_symbol,
                    )
                    if results:
                        for result in results:
                            try:
                                scraped_data = await self.crawler.arun(result["url"])
                                if scraped_data and scraped_data.markdown:
                                    article = NewsArticle(
                                        title=result.get("title", ""),
                                        url=result.get("url", ""),
                                        content=scraped_data.markdown,
                                        summary=generate_summary(scraped_data.markdown),
                                        source=result.get("source", ""),
                                        published_at=result.get("date", ""),
                                        symbols=[symbol],
                                    )
                                    all_articles.append(article)
                            except Exception as e:
                                logger.warning(f"Failed to scrape article {result['url']}: {e}")
                except Exception as e:
                    logger.error(f"Failed to search for news for symbol {symbol}: {e}")
        return all_articles

    async def search_and_scrape_financial_reports(
        self, company_symbol: str, report_type: str = "10-K"
    ) -> List[Dict[str, Any]]:
        query = f"{company_symbol} {report_type} site:sec.gov"
        filings = []
        async with DDGS() as ddgs:
            try:
                results = await ddgs.text(
                    keywords=query,
                    region="us-en",
                    safesearch="off",
                    max_results=5,
                )
                if results:
                    for result in results:
                        try:
                            scraped_data = await self.crawler.arun(result["href"])
                            if scraped_data and scraped_data.markdown:
                                filings.append(
                                    {
                                        "url": result["href"],
                                        "content": scraped_data.markdown,
                                        "report_type": report_type,
                                    }
                                )
                        except Exception as e:
                            logger.warning(f"Failed to scrape report {result['href']}: {e}")
            except Exception as e:
                logger.error(f"Failed to search for financial reports for symbol {company_symbol}: {e}")
        return filings

    async def start(self):
        logger.info("LocalScrapingService started.")
        await self.crawler.start()

    async def stop(self):
        logger.info("LocalScrapingService stopped.")
        await self.crawler.close()

    async def health_check(self) -> bool:
        # For now, we assume the service is healthy if it can be instantiated.
        return True
