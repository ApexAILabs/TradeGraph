import pytest
from unittest.mock import patch
from datetime import datetime

from radgegraph_financial_advisor.agents.base_agent import BaseAgent
from radgegraph_financial_advisor.agents.news_agent import NewsReaderAgent
from radgegraph_financial_advisor.agents.report_analysis_agent import (
    ReportAnalysisAgent,
)
from radgegraph_financial_advisor.agents.recommendation_engine import (
    TradingRecommendationEngine,
)
from radgegraph_financial_advisor.agents.financial_agent import FinancialAnalysisAgent
from radgegraph_financial_advisor.models.financial_data import (
    NewsArticle,
    SentimentType,
)


class TestBaseAgent:
    """Test base agent functionality."""

    class ConcreteAgent(BaseAgent):
        async def execute(self, input_data):
            return {"test": "result"}

    @pytest.mark.asyncio
    async def test_agent_lifecycle(self):
        """Test agent start/stop lifecycle."""
        agent = self.ConcreteAgent("test-agent", "Test agent")

        assert not agent.is_active
        assert agent.name == "test-agent"
        assert agent.description == "Test agent"

        await agent.start()
        assert agent.is_active

        await agent.stop()
        assert not agent.is_active

    @pytest.mark.asyncio
    async def test_agent_status(self):
        """Test agent status reporting."""
        agent = self.ConcreteAgent("test-agent", "Test agent")
        status = agent.get_status()

        assert status["name"] == "test-agent"
        assert status["description"] == "Test agent"
        assert "created_at" in status
        assert "last_activity" in status
        assert "is_active" in status

    @pytest.mark.asyncio
    async def test_health_check(self):
        """Test agent health check."""
        agent = self.ConcreteAgent("test-agent", "Test agent")
        health_ok = await agent.health_check()
        assert health_ok is True


class TestNewsReaderAgent:
    """Test NewsReaderAgent functionality."""

    @pytest.mark.asyncio
    async def test_news_agent_initialization(self):
        """Test news agent initialization."""
        agent = NewsReaderAgent()
        assert agent.name == "NewsReaderAgent"
        assert "financial news" in agent.description.lower()

    @pytest.mark.asyncio
    async def test_news_agent_lifecycle(self, mock_aiohttp_session):
        """Test news agent start/stop with session management."""
        agent = NewsReaderAgent()

        with patch("aiohttp.ClientSession", return_value=mock_aiohttp_session):
            await agent.start()
            assert agent.is_active
            assert agent.session is not None

            await agent.stop()
            assert not agent.is_active

    @pytest.mark.asyncio
    async def test_execute_news_collection(
        self, mock_aiohttp_session, sample_news_articles
    ):
        """Test news collection execution."""
        agent = NewsReaderAgent()

        # Mock the session and article extraction
        with patch("aiohttp.ClientSession", return_value=mock_aiohttp_session):
            with patch.object(agent, "_fetch_news_from_source") as mock_fetch:
                mock_fetch.return_value = [
                    NewsArticle(**article) for article in sample_news_articles
                ]

                await agent.start()

                input_data = {
                    "symbols": ["AAPL", "MSFT"],
                    "timeframe_hours": 24,
                    "max_articles": 10,
                }

                result = await agent.execute(input_data)

                assert "articles" in result
                assert "total_count" in result
                assert result["total_count"] > 0

                await agent.stop()

    @pytest.mark.asyncio
    async def test_sentiment_analysis(self):
        """Test sentiment analysis functionality."""
        agent = NewsReaderAgent()

        # Test bullish sentiment
        bullish_content = (
            "Apple reports strong growth and profit gains with positive outlook"
        )
        sentiment = await agent._analyze_sentiment(bullish_content)
        assert sentiment == SentimentType.BULLISH

        # Test bearish sentiment
        bearish_content = (
            "Company faces loss and decline with negative weak performance"
        )
        sentiment = await agent._analyze_sentiment(bearish_content)
        assert sentiment == SentimentType.BEARISH

        # Test neutral sentiment
        neutral_content = "Company reports standard quarterly results"
        sentiment = await agent._analyze_sentiment(neutral_content)
        assert sentiment == SentimentType.NEUTRAL

    @pytest.mark.asyncio
    async def test_impact_score_calculation(self):
        """Test impact score calculation."""
        agent = NewsReaderAgent()

        # Create test article
        article = NewsArticle(
            title="Apple Earnings Beat Expectations",
            url="https://test.com",
            content="Apple Inc. reported strong earnings with revenue growth",
            source="test",
            published_at=datetime.now(),
            symbols=["AAPL"],
        )

        symbols = ["AAPL"]
        impact_score = await agent._calculate_impact_score(article, symbols)

        assert 0.0 <= impact_score <= 1.0
        assert impact_score > 0.5  # Should be higher due to symbol mention in title


class TestReportAnalysisAgent:
    """Test ReportAnalysisAgent functionality."""

    @pytest.mark.asyncio
    async def test_report_agent_initialization(self, mock_local_scraping_service):
        """Test report agent initialization."""
        agent = ReportAnalysisAgent(scraping_service=mock_local_scraping_service)
        assert agent.name == "ReportAnalysisAgent"
        assert "report" in agent.description.lower()

    @pytest.mark.asyncio
    async def test_execute_report_analysis(
        self, mock_local_scraping_service, mock_langchain_llm
    ):
        """Test report analysis execution."""
        agent = ReportAnalysisAgent(scraping_service=mock_local_scraping_service)
        agent.llm = mock_langchain_llm

        mock_local_scraping_service.search_and_scrape_financial_reports.return_value = [
            {
                "url": "https://sec.gov/test-filing",
                "content": "Sample 10-K filing content with financial data",
                "report_type": "10-K",
            }
        ]

        await agent.start()

        input_data = {
            "symbols": ["AAPL"],
            "report_types": ["10-K"],
            "analysis_depth": "detailed",
        }

        result = await agent.execute(input_data)

        assert "report_analysis" in result
        assert "AAPL" in result["report_analysis"]

        await agent.stop()


class _DummyFinnhubClient:
    async def get_quote(self, symbol):
        return {"c": 101.0, "o": 99.0, "v": 1250000}

    async def get_company_profile(self, symbol):
        return {"name": f"{symbol} Inc.", "marketCapitalization": 1_250_000_000}

    async def get_candles(self, symbol, *, resolution, start, end):
        closes = [float(90 + i) for i in range(160)]
        return {
            "s": "ok",
            "c": closes,
            "h": [price + 1 for price in closes],
            "l": [price - 1 for price in closes],
        }

    async def close(self):
        return None


class _DummyBinanceClient:
    async def close(self):
        return None


class _DummyAlphaClient:
    async def close(self):
        return None

    async def get_daily_time_series(self, symbol, **_):
        return {
            "timestamp": "2024-01-02",
            "close": 101.0,
            "open": 100.0,
            "high": 102.0,
            "low": 99.0,
            "volume": 1000000,
        }

    async def get_intraday_time_series(self, symbol, **_):
        return {
            "timestamp": "2024-01-02 15:30:00",
            "close": 100.5,
            "open": 100.2,
            "high": 100.8,
            "low": 100.1,
            "volume": 10000,
        }

    async def get_technical_indicator(self, symbol, *, indicator, **_):
        return {
            "indicator": indicator,
            "timestamp": "2024-01-02 15:30:00",
            "values": {indicator: 100.25},
        }

    async def get_company_overview(self, symbol):
        return {
            "symbol": symbol,
            "name": "Demo Corp",
            "sector": "Technology",
            "market_cap": 1500000000,
            "pe_ratio": 24.5,
            "eps": 6.2,
            "revenue": 50000000000,
            "dividend_yield": 0.012,
            "return_on_equity": 0.18,
            "return_on_assets": 0.11,
            "debt_to_equity": 0.45,
            "current_ratio": 1.2,
            "price_to_book": 5.1,
            "beta": 1.1,
            "fifty_two_week_high": 120.0,
            "fifty_two_week_low": 80.0,
        }

    async def get_sector_performance(self):
        return {"Rank A: Real-Time Performance": {"Technology": "+1.2%"}}

    async def get_fx_rate(self, from_symbol, to_symbol):
        return {
            "from_symbol": from_symbol,
            "to_symbol": to_symbol,
            "exchange_rate": 1.083,
            "last_refreshed": "2024-01-02 15:30:00",
        }

    async def get_crypto_rate(self, symbol, market):
        return {
            "from_symbol": symbol,
            "to_symbol": market,
            "exchange_rate": 45250.5,
            "last_refreshed": "2024-01-02 15:30:00",
        }


@pytest.mark.asyncio
async def test_financial_agent_collects_alpha_vantage_data():
    agent = FinancialAnalysisAgent(
        finnhub_client=_DummyFinnhubClient(),
        binance_client=_DummyBinanceClient(),
        alpha_vantage_client=_DummyAlphaClient(),
    )

    payload = {
        "symbols": ["AAPL"],
        "include_financials": True,
        "include_market_data": True,
        "include_technical": False,
        "alpha_vantage_requests": {
            "datasets": [
                "daily",
                "intraday",
                "technical",
                "fundamentals",
                "sector",
                "fx",
                "crypto",
            ],
            "intraday_interval": "5min",
            "technical_indicators": ["SMA"],
            "fx_pairs": ["EUR/USD"],
            "crypto_pairs": ["BTC/USD"],
        },
    }

    result = await agent.execute(payload)

    alpha_summary = result.get("alpha_vantage", {})
    assert "AAPL" in alpha_summary.get("per_symbol", {})
    assert alpha_summary["per_symbol"]["AAPL"]["daily"]["close"] == pytest.approx(101.0)
    assert alpha_summary["global"]["fx_quotes"][0]["to_symbol"] == "USD"

    analysis = result["analysis_results"]["AAPL"]
    assert analysis["financials"]["company_name"] == "Demo Corp"
    assert (
        analysis["alpha_vantage"]["technical_indicators"]["SMA"]["indicator"] == "SMA"
    )


class TestTradingRecommendationEngine:
    """Test TradingRecommendationEngine functionality."""

    @pytest.mark.asyncio
    async def test_recommendation_engine_initialization(self, mock_langchain_llm):
        """Test recommendation engine initialization."""
        with patch(
            "radgegraph_financial_advisor.agents.recommendation_engine.ChatOpenAI",
            return_value=mock_langchain_llm,
        ):
            engine = TradingRecommendationEngine()
            assert engine.name == "TradingRecommendationEngine"
            assert "recommendation" in engine.description.lower()

    @pytest.mark.asyncio
    async def test_execute_recommendation_generation(
        self, mock_langchain_llm, sample_analysis_context
    ):
        """Test recommendation generation execution."""
        with patch(
            "radgegraph_financial_advisor.agents.recommendation_engine.ChatOpenAI",
            return_value=mock_langchain_llm,
        ):
            engine = TradingRecommendationEngine()

            await engine.start()

            input_data = {
                "analysis_contexts": sample_analysis_context,
                "portfolio_constraints": {"portfolio_size": 100000, "max_positions": 5},
                "risk_preferences": {"risk_tolerance": "medium"},
            }

            result = await engine.execute(input_data)

            assert "individual_recommendations" in result
            assert "portfolio_recommendation" in result
            assert "alerts" in result

            await engine.stop()

    @pytest.mark.asyncio
    async def test_fundamental_score_calculation(
        self, mock_langchain_llm, sample_financial_data
    ):
        """Test fundamental score calculation."""
        with patch(
            "radgegraph_financial_advisor.agents.recommendation_engine.ChatOpenAI",
            return_value=mock_langchain_llm,
        ):
            engine = TradingRecommendationEngine()

            financials = sample_financial_data["AAPL"]
            report_analysis = {"financial_health_score": 8.5}

            score = await engine._calculate_fundamental_score(
                financials, report_analysis
            )

            assert 0.0 <= score <= 1.0
            assert score > 0.5  # Should be positive for good financials

    @pytest.mark.asyncio
    async def test_technical_score_calculation(
        self, mock_langchain_llm, sample_technical_data, sample_market_data
    ):
        """Test technical score calculation."""
        with patch(
            "radgegraph_financial_advisor.agents.recommendation_engine.ChatOpenAI",
            return_value=mock_langchain_llm,
        ):
            engine = TradingRecommendationEngine()

            technical_data = sample_technical_data["AAPL"]
            market_data = sample_market_data["AAPL"]

            score = await engine._calculate_technical_score(technical_data, market_data)

            assert 0.0 <= score <= 1.0

    @pytest.mark.asyncio
    async def test_sentiment_score_calculation(self, mock_langchain_llm):
        """Test sentiment score calculation."""
        with patch(
            "radgegraph_financial_advisor.agents.recommendation_engine.ChatOpenAI",
            return_value=mock_langchain_llm,
        ):
            engine = TradingRecommendationEngine()

            # Test positive sentiment
            positive_sentiment = {"sentiment_score": 0.5, "confidence": 0.8}
            score = await engine._calculate_sentiment_score(positive_sentiment)
            assert score > 0.5

            # Test negative sentiment
            negative_sentiment = {"sentiment_score": -0.5, "confidence": 0.8}
            score = await engine._calculate_sentiment_score(negative_sentiment)
            assert score < 0.5

    @pytest.mark.asyncio
    async def test_risk_level_calculation(self, mock_langchain_llm):
        """Test risk level calculation."""
        with patch(
            "radgegraph_financial_advisor.agents.recommendation_engine.ChatOpenAI",
            return_value=mock_langchain_llm,
        ):
            engine = TradingRecommendationEngine()

            # Low risk scenario
            low_risk_data = {
                "market_data": {},
                "financials": {"beta": 0.8, "debt_to_equity": 0.3, "market_cap": 1e12},
                "technical_data": {"rsi": 50},
                "report_analysis": {"risk_factors": []},
            }

            risk_level = engine._calculate_risk_level(
                low_risk_data["market_data"],
                low_risk_data["financials"],
                low_risk_data["technical_data"],
                low_risk_data["report_analysis"],
            )

            assert risk_level in ["low", "medium", "high", "very_high"]

    @pytest.mark.asyncio
    async def test_position_size_calculation(self, mock_langchain_llm):
        """Test position size calculation."""
        with patch(
            "radgegraph_financial_advisor.agents.recommendation_engine.ChatOpenAI",
            return_value=mock_langchain_llm,
        ):
            engine = TradingRecommendationEngine()

            from radgegraph_financial_advisor.models.recommendations import (
                RiskLevel,
                RecommendationType,
            )

            # Test different confidence and risk combinations
            test_cases = [
                (
                    0.9,
                    RiskLevel.LOW,
                    {"risk_tolerance": "aggressive"},
                    RecommendationType.BUY,
                ),
                (
                    0.5,
                    RiskLevel.MEDIUM,
                    {"risk_tolerance": "medium"},
                    RecommendationType.HOLD,
                ),
                (
                    0.3,
                    RiskLevel.HIGH,
                    {"risk_tolerance": "conservative"},
                    RecommendationType.SELL,
                ),
            ]

            for confidence, risk_level, risk_prefs, recommendation_type in test_cases:
                position_size = engine._calculate_position_size(
                    confidence, risk_level, risk_prefs, recommendation_type
                )
                assert position_size > 0.01

    @pytest.mark.asyncio
    async def test_price_targets_calculation(self, mock_langchain_llm):
        """Test price target calculation."""
        with patch(
            "radgegraph_financial_advisor.agents.recommendation_engine.ChatOpenAI",
            return_value=mock_langchain_llm,
        ):
            engine = TradingRecommendationEngine()

            from radgegraph_financial_advisor.models.recommendations import (
                RecommendationType,
            )

            current_price = 195.0
            technical_data = {"resistance_level": 210.0, "support_level": 180.0}
            financials = {"pe_ratio": 25.0}

            target_price, stop_loss = await engine._calculate_price_targets(
                "AAPL",
                current_price,
                RecommendationType.BUY,
                technical_data,
                financials,
            )

            if target_price:
                assert target_price > current_price  # Target should be higher for BUY
            if stop_loss:
                assert stop_loss < current_price  # Stop loss should be lower for BUY
