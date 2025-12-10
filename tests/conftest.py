import asyncio
import pytest
import os
from unittest.mock import AsyncMock
from datetime import datetime, timedelta

from tradegraph_financial_advisor.models.financial_data import NewsArticle

# Test configuration
os.environ["OPENAI_API_KEY"] = "test-openai-key"
os.environ["ALPHA_VANTAGE_API_KEY"] = "test-alpha-vantage-key"
os.environ["FINNHUB_API_KEY"] = "test-finnhub-key"
os.environ["LOG_LEVEL"] = "DEBUG"


@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def mock_openai_response():
    """Mock OpenAI API response."""
    mock_response = AsyncMock()
    mock_response.content = '{"sentiment_score": 0.2, "sentiment_label": "bullish", "confidence": 0.8, "key_themes": ["growth", "innovation"], "sentiment_drivers": ["strong earnings", "new product launch"]}'
    return mock_response


@pytest.fixture
def sample_news_articles():
    """Sample news articles for testing."""
    return [
        {
            "title": "Apple Reports Strong Q4 Earnings",
            "url": "https://example.com/apple-earnings",
            "content": "Apple Inc. reported strong fourth quarter earnings with revenue growth of 15%...",
            "source": "reuters",
            "published_at": datetime.now().isoformat(),
            "symbols": ["AAPL"],
            "sentiment": "bullish",
            "impact_score": 0.8,
        },
        {
            "title": "Microsoft Azure Growth Continues",
            "url": "https://example.com/msft-azure",
            "content": "Microsoft's Azure cloud platform continues to show strong growth...",
            "source": "bloomberg",
            "published_at": (datetime.now() - timedelta(hours=2)).isoformat(),
            "symbols": ["MSFT"],
            "sentiment": "bullish",
            "impact_score": 0.7,
        },
    ]


@pytest.fixture
def sample_market_data():
    """Sample market data for testing."""
    return {
        "AAPL": {
            "symbol": "AAPL",
            "current_price": 195.89,
            "change": 2.15,
            "change_percent": 1.11,
            "volume": 45234567,
            "market_cap": 3000000000000,
            "pe_ratio": 28.5,
            "timestamp": datetime.now().isoformat(),
        },
        "MSFT": {
            "symbol": "MSFT",
            "current_price": 374.50,
            "change": -1.25,
            "change_percent": -0.33,
            "volume": 23456789,
            "market_cap": 2800000000000,
            "pe_ratio": 32.1,
            "timestamp": datetime.now().isoformat(),
        },
    }


@pytest.fixture
def sample_financial_data():
    """Sample financial data for testing."""
    return {
        "AAPL": {
            "symbol": "AAPL",
            "company_name": "Apple Inc.",
            "market_cap": 3000000000000,
            "pe_ratio": 28.5,
            "eps": 6.88,
            "revenue": 394328000000,
            "net_income": 99803000000,
            "debt_to_equity": 1.73,
            "current_ratio": 1.05,
            "return_on_equity": 0.175,
            "return_on_assets": 0.225,
            "price_to_book": 5.02,
            "dividend_yield": 0.0047,
            "beta": 1.29,
            "fifty_two_week_high": 199.62,
            "fifty_two_week_low": 164.08,
            "current_price": 195.89,
            "report_date": datetime.now().isoformat(),
            "report_type": "quarterly",
        }
    }


@pytest.fixture
def sample_technical_data():
    """Sample technical indicators for testing."""
    return {
        "AAPL": {
            "symbol": "AAPL",
            "sma_20": 185.50,
            "sma_50": 180.25,
            "ema_12": 190.15,
            "ema_26": 187.80,
            "rsi": 65.2,
            "macd": 2.35,
            "macd_signal": 1.85,
            "bollinger_upper": 198.50,
            "bollinger_lower": 175.25,
            "support_level": 185.00,
            "resistance_level": 200.00,
            "timestamp": datetime.now().isoformat(),
        }
    }


@pytest.fixture
def sample_recommendations():
    """Sample trading recommendations for testing."""
    return [
        {
            "symbol": "AAPL",
            "company_name": "Apple Inc.",
            "recommendation": "buy",
            "confidence_score": 0.785,
            "target_price": 225.00,
            "stop_loss": 175.00,
            "current_price": 195.89,
            "risk_level": "medium",
            "time_horizon": "medium_term",
            "recommended_allocation": 0.085,
            "fundamental_score": 0.80,
            "technical_score": 0.75,
            "sentiment_score": 0.80,
            "key_factors": ["Strong iPhone sales", "Services growth", "AI integration"],
            "risks": ["Market competition", "Regulatory concerns"],
            "catalysts": ["iPhone 16 launch", "Vision Pro adoption"],
            "analyst_notes": "Strong fundamentals with positive growth outlook",
            "sector": "Technology",
            "expected_return": 0.15,
        },
        {
            "symbol": "MSFT",
            "company_name": "Microsoft Corporation",
            "recommendation": "strong_buy",
            "confidence_score": 0.852,
            "target_price": 420.00,
            "stop_loss": 340.00,
            "current_price": 374.50,
            "risk_level": "low",
            "time_horizon": "long_term",
            "recommended_allocation": 0.12,
            "fundamental_score": 0.85,
            "technical_score": 0.80,
            "sentiment_score": 0.90,
            "key_factors": [
                "Azure growth",
                "AI leadership",
                "Strong enterprise business",
            ],
            "risks": ["Cloud competition", "Economic slowdown"],
            "catalysts": ["AI monetization", "Copilot adoption"],
            "analyst_notes": "Dominant position in cloud and AI markets",
            "sector": "Technology",
            "expected_return": 0.12,
        },
    ]


@pytest.fixture
def sample_channel_streams():
    return {
        "top_market_crypto": {
            "channel_id": "top_market_crypto",
            "title": "Top Market & Crypto Headlines",
            "items": [
                {
                    "title": "Markets rally on soft CPI",
                    "summary": "Major indices advanced after inflation cooled.",
                    "source": "Reuters",
                    "published_at": datetime.now().isoformat(),
                },
                {
                    "title": "Bitcoin holds above 60k",
                    "summary": "Crypto markets consolidate gains.",
                    "source": "CoinDesk",
                    "published_at": datetime.now().isoformat(),
                },
            ],
        },
        "open_source_agencies": {
            "channel_id": "open_source_agencies",
            "title": "Open News Agencies",
            "items": [
                {
                    "title": "Guardian: policy outlook improves",
                    "summary": "Central banks signal patience.",
                    "source": "Guardian",
                    "published_at": datetime.now().isoformat(),
                }
            ],
        },
    }


@pytest.fixture
def sample_price_trends():
    return {
        "AAPL": {
            "symbol": "AAPL",
            "current_price": 195.0,
            "pricing_timestamp": datetime.now().isoformat(),
            "trends": {
                "last_month": {
                    "start": 180.0,
                    "end": 195.0,
                    "change": 15.0,
                    "percent_change": 8.3,
                    "direction": "bullish",
                },
                "last_week": {
                    "start": 190.0,
                    "end": 195.0,
                    "change": 5.0,
                    "percent_change": 2.6,
                    "direction": "bullish",
                },
                "last_day": {
                    "start": 194.0,
                    "end": 195.0,
                    "change": 1.0,
                    "percent_change": 0.5,
                    "direction": "bullish",
                },
                "last_hour": {
                    "start": 194.5,
                    "end": 195.0,
                    "change": 0.5,
                    "percent_change": 0.26,
                    "direction": "bullish",
                },
            },
        }
    }


@pytest.fixture
def sample_portfolio_recommendation():
    """Sample portfolio recommendation for testing."""
    return {
        "recommendations": [
            {
                "symbol": "AAPL",
                "recommendation": "buy",
                "confidence_score": 0.785,
                "recommended_allocation": 0.085,
            },
            {
                "symbol": "MSFT",
                "recommendation": "strong_buy",
                "confidence_score": 0.852,
                "recommended_allocation": 0.12,
            },
        ],
        "total_confidence": 0.8185,
        "diversification_score": 0.75,
        "overall_risk_level": "medium",
        "portfolio_size": 100000,
        "sector_weights": {"Technology": 0.205},
        "rebalancing_frequency": "quarterly",
    }


@pytest.fixture
def mock_langchain_llm():
    """Mock LangChain LLM for testing."""
    mock_llm = AsyncMock()
    mock_llm.ainvoke.return_value = AsyncMock(
        content='{"recommendation": "buy", "confidence_score": 0.8, "target_price": 220.0}'
    )
    return mock_llm


class MockHTTPResponse:
    """Mock HTTP response for testing."""

    def __init__(self, status=200, json_data=None, text_data=None):
        self.status = status
        self._json_data = json_data or {}
        self._text_data = text_data or ""

    async def json(self):
        return self._json_data

    async def text(self):
        return self._text_data

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        pass


@pytest.fixture
def mock_aiohttp_session():
    """Mock aiohttp session for testing."""
    mock_session = AsyncMock()
    mock_session.get.return_value = MockHTTPResponse(
        status=200,
        json_data={"test": "data"},
        text_data="<html><body>Test content</body></html>",
    )
    mock_session.post.return_value = MockHTTPResponse(
        status=200, json_data={"success": True}
    )
    mock_session.close.return_value = None
    return mock_session


@pytest.fixture
def sample_analysis_context():
    """Sample analysis context for testing."""
    return {
        "AAPL": {
            "symbol": "AAPL",
            "news_articles": [
                {
                    "title": "Apple Earnings Beat",
                    "content": "Strong quarterly results",
                    "sentiment": "bullish",
                }
            ],
            "market_data": {"current_price": 195.89, "volume": 45000000},
            "technical_indicators": {"rsi": 65.2, "sma_20": 185.50},
            "sentiment_analysis": {"sentiment_score": 0.2, "confidence": 0.8},
        }
    }


# Test data helpers
def create_test_news_article(symbol="AAPL", sentiment="bullish"):
    """Helper to create test news article."""
    return {
        "title": f"{symbol} Test News",
        "url": f"https://example.com/{symbol.lower()}-news",
        "content": f"Test news content for {symbol}",
        "source": "test-source",
        "published_at": datetime.now().isoformat(),
        "symbols": [symbol],
        "sentiment": sentiment,
        "impact_score": 0.7,
    }


def create_test_recommendation(symbol="AAPL", recommendation="buy"):
    """Helper to create test recommendation."""
    return {
        "symbol": symbol,
        "company_name": f"{symbol} Inc.",
        "recommendation": recommendation,
        "confidence_score": 0.8,
        "target_price": 200.0,
        "current_price": 195.0,
        "risk_level": "medium",
        "time_horizon": "medium_term",
        "recommended_allocation": 0.1,
        "fundamental_score": 0.8,
        "technical_score": 0.75,
        "sentiment_score": 0.85,
    }


@pytest.fixture
def mock_local_scraping_service():
    """Mock LocalScrapingService for testing."""
    mock_service = AsyncMock()
    mock_service.start.return_value = None
    mock_service.stop.return_value = None
    mock_service.search_and_scrape_news.return_value = [
        NewsArticle(
            title="Sample Scraped News",
            url="https://example.com/scraped-news",
            content="Content from scraped news.",
            source="local-scraper",
            published_at=datetime.now().isoformat(),
            symbols=["AAPL"],
        )
    ]
    mock_service.search_and_scrape_financial_reports.return_value = [
        {"url": "https://example.com/scraped-report", "content": "10K filing content"}
    ]
    mock_service.health_check.return_value = True
    return mock_service
