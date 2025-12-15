from typing import Any, Dict, List, Optional
from datetime import datetime
import aiohttp
from bs4 import BeautifulSoup
from loguru import logger

from .base_agent import BaseAgent
from ..models.financial_data import NewsArticle, SentimentType
from ..config.settings import settings
from ..utils.helpers import generate_summary


class NewsReaderAgent(BaseAgent):
    def __init__(self, **kwargs):
        super().__init__(
            name="NewsReaderAgent",
            description="Reads and analyzes financial news from multiple sources",
            **kwargs,
        )
        self.session: Optional[aiohttp.ClientSession] = None

    async def start(self) -> None:
        await super().start()
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=settings.analysis_timeout_seconds)
        )

    async def stop(self) -> None:
        if self.session:
            await self.session.close()
        await super().stop()

    async def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        symbols = input_data.get("symbols", [])
        timeframe_hours = input_data.get("timeframe_hours", 24)
        max_articles = input_data.get("max_articles", 50)

        logger.info(f"Fetching news for symbols: {symbols}")

        all_articles = []

        for source in settings.news_sources:
            try:
                articles = await self._fetch_news_from_source(
                    source,
                    symbols,
                    timeframe_hours,
                    max_articles // len(settings.news_sources),
                )
                all_articles.extend(articles)
            except Exception as e:
                logger.error(f"Failed to fetch news from {source}: {str(e)}")

        analyzed_articles = await self._analyze_articles(all_articles, symbols)

        return {
            "articles": [article.model_dump() for article in analyzed_articles],
            "total_count": len(analyzed_articles),
            "sources": settings.news_sources,
            "analysis_timestamp": datetime.now().isoformat(),
        }

    async def _fetch_news_from_source(
        self, source: str, symbols: List[str], timeframe_hours: int, max_articles: int
    ) -> List[NewsArticle]:
        articles = []

        if source == "yahoo-finance":
            articles = await self._fetch_yahoo_finance_news(
                symbols, timeframe_hours, max_articles
            )
        elif source == "bloomberg":
            articles = await self._fetch_bloomberg_news(
                symbols, timeframe_hours, max_articles
            )
        elif source == "reuters":
            articles = await self._fetch_reuters_news(
                symbols, timeframe_hours, max_articles
            )
        elif source == "marketwatch":
            articles = await self._fetch_marketwatch_news(
                symbols, timeframe_hours, max_articles
            )
        elif source == "cnbc":
            articles = await self._fetch_cnbc_news(
                symbols, timeframe_hours, max_articles
            )

        return articles

    async def _fetch_yahoo_finance_news(
        self, symbols: List[str], timeframe_hours: int, max_articles: int
    ) -> List[NewsArticle]:
        articles = []

        for symbol in symbols:
            try:
                url = f"https://finance.yahoo.com/quote/{symbol}/news"
                async with self.session.get(url) as response:
                    if response.status == 200:
                        html = await response.text()
                        soup = BeautifulSoup(html, "html.parser")

                        news_items = soup.find_all(
                            "div", class_=["Mb(5px)", "news-item"]
                        )[:max_articles]

                        for item in news_items:
                            try:
                                title_elem = item.find("h3") or item.find("a")
                                if title_elem:
                                    title = title_elem.get_text(strip=True)
                                    link = title_elem.get("href", "")
                                    if link and not link.startswith("http"):
                                        link = f"https://finance.yahoo.com{link}"

                                    # Extract article content
                                    raw_content = await self._extract_article_content(
                                        link
                                    )
                                    content = (raw_content or "")[
                                        :1000
                                    ]  # Limit content length

                                    article = NewsArticle(
                                        title=title,
                                        url=link,
                                        content=content,
                                        summary=generate_summary(raw_content or title),
                                        source="yahoo-finance",
                                        published_at=datetime.now(),
                                        symbols=[symbol],
                                    )
                                    articles.append(article)
                            except Exception as e:
                                logger.warning(
                                    f"Error parsing Yahoo Finance news item: {str(e)}"
                                )

            except Exception as e:
                logger.error(
                    f"Error fetching Yahoo Finance news for {symbol}: {str(e)}"
                )

        return articles

    async def _fetch_bloomberg_news(
        self, symbols: List[str], timeframe_hours: int, max_articles: int
    ) -> List[NewsArticle]:
        # Bloomberg typically requires subscription, so this is a simplified version
        # In production, you'd use Bloomberg API or authorized scraping
        articles = []

        try:
            search_query = " OR ".join(symbols)
            url = f"https://www.bloomberg.com/search?query={search_query}"

            async with self.session.get(url) as response:
                if response.status == 200:
                    html = await response.text()
                    soup = BeautifulSoup(html, "html.parser")

                    # Extract news items (this is simplified)
                    news_items = soup.find_all("div", class_="storyItem")[:max_articles]

                    for item in news_items:
                        try:
                            title_elem = item.find("a")
                            if title_elem:
                                title = title_elem.get_text(strip=True)
                                link = title_elem.get("href", "")
                                if link and not link.startswith("http"):
                                    link = f"https://www.bloomberg.com{link}"

                                raw_content = await self._extract_article_content(link)
                                content = (raw_content or "")[:1000]

                                article = NewsArticle(
                                    title=title,
                                    url=link,
                                    content=content,
                                    summary=generate_summary(raw_content or title),
                                    source="bloomberg",
                                    published_at=datetime.now(),
                                    symbols=symbols,
                                )
                                articles.append(article)
                        except Exception as e:
                            logger.warning(
                                f"Error parsing Bloomberg news item: {str(e)}"
                            )

        except Exception as e:
            logger.error(f"Error fetching Bloomberg news: {str(e)}")

        return articles

    async def _fetch_reuters_news(
        self, symbols: List[str], timeframe_hours: int, max_articles: int
    ) -> List[NewsArticle]:
        articles = []

        try:
            for symbol in symbols:
                url = f"https://www.reuters.com/markets/companies/{symbol}/"

                async with self.session.get(url) as response:
                    if response.status == 200:
                        html = await response.text()
                        soup = BeautifulSoup(html, "html.parser")

                        news_items = soup.find_all("div", class_="story-card")[
                            :max_articles
                        ]

                        for item in news_items:
                            try:
                                title_elem = item.find("a")
                                if title_elem:
                                    title = title_elem.get_text(strip=True)
                                    link = title_elem.get("href", "")
                                    if link and not link.startswith("http"):
                                        link = f"https://www.reuters.com{link}"

                                    raw_content = await self._extract_article_content(
                                        link
                                    )
                                    content = (raw_content or "")[:1000]

                                    article = NewsArticle(
                                        title=title,
                                        url=link,
                                        content=content,
                                        summary=generate_summary(raw_content or title),
                                        source="reuters",
                                        published_at=datetime.now(),
                                        symbols=[symbol],
                                    )
                                    articles.append(article)
                            except Exception as e:
                                logger.warning(
                                    f"Error parsing Reuters news item: {str(e)}"
                                )

        except Exception as e:
            logger.error(f"Error fetching Reuters news: {str(e)}")

        return articles

    async def _fetch_marketwatch_news(
        self, symbols: List[str], timeframe_hours: int, max_articles: int
    ) -> List[NewsArticle]:
        # Similar implementation for MarketWatch
        return []

    async def _fetch_cnbc_news(
        self, symbols: List[str], timeframe_hours: int, max_articles: int
    ) -> List[NewsArticle]:
        # Similar implementation for CNBC
        return []

    async def _extract_article_content(self, url: str) -> str:
        try:
            if not self.session:
                return ""

            async with self.session.get(url) as response:
                if response.status == 200:
                    html = await response.text()
                    soup = BeautifulSoup(html, "html.parser")

                    # Remove script and style elements
                    for script in soup(["script", "style"]):
                        script.decompose()

                    # Get text content
                    text = soup.get_text()

                    # Clean up whitespace
                    lines = (line.strip() for line in text.splitlines())
                    chunks = (
                        phrase.strip() for line in lines for phrase in line.split("  ")
                    )
                    text = " ".join(chunk for chunk in chunks if chunk)

                    return text[:2000]  # Limit content length

        except Exception as e:
            logger.warning(f"Error extracting content from {url}: {str(e)}")
            return ""

    async def _analyze_articles(
        self, articles: List[NewsArticle], symbols: List[str]
    ) -> List[NewsArticle]:
        analyzed_articles = []

        for article in articles:
            try:
                # Add sentiment analysis
                sentiment = await self._analyze_sentiment(article.content)
                article.sentiment = sentiment

                # Add impact score
                impact_score = await self._calculate_impact_score(article, symbols)
                article.impact_score = impact_score

                if not article.summary:
                    article.summary = generate_summary(article.content or article.title)

                analyzed_articles.append(article)

            except Exception as e:
                logger.warning(f"Error analyzing article: {str(e)}")
                analyzed_articles.append(article)

        return analyzed_articles

    async def _analyze_sentiment(self, content: str) -> SentimentType:
        # Simple keyword-based sentiment analysis
        # In production, use a proper NLP model

        bullish_keywords = [
            "buy",
            "bullish",
            "positive",
            "growth",
            "profit",
            "strong",
            "gains",
            "upgrade",
        ]
        bearish_keywords = [
            "sell",
            "bearish",
            "negative",
            "loss",
            "decline",
            "weak",
            "downgrade",
            "risk",
        ]

        content_lower = content.lower()

        bullish_count = sum(
            1 for keyword in bullish_keywords if keyword in content_lower
        )
        bearish_count = sum(
            1 for keyword in bearish_keywords if keyword in content_lower
        )

        if bullish_count > bearish_count:
            return SentimentType.BULLISH
        elif bearish_count > bullish_count:
            return SentimentType.BEARISH
        else:
            return SentimentType.NEUTRAL

    async def _calculate_impact_score(
        self, article: NewsArticle, symbols: List[str]
    ) -> float:
        # Calculate impact score based on various factors
        score = 0.5  # Base score

        # Boost score if article mentions specific symbols
        content_lower = article.content.lower()
        title_lower = article.title.lower()

        for symbol in symbols:
            if symbol.lower() in title_lower:
                score += 0.2
            elif symbol.lower() in content_lower:
                score += 0.1

        # Boost score for high-impact keywords
        high_impact_keywords = [
            "earnings",
            "merger",
            "acquisition",
            "partnership",
            "lawsuit",
            "fda approval",
        ]
        for keyword in high_impact_keywords:
            if keyword in content_lower or keyword in title_lower:
                score += 0.15

        return min(score, 1.0)

    async def _health_check_impl(self) -> None:
        if self.session and not self.session.closed:
            # Simple health check by making a request
            async with self.session.get("https://httpbin.org/status/200") as response:
                if response.status != 200:
                    raise Exception("Health check failed")
