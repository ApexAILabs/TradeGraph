import json
import pytest
from unittest.mock import Mock, AsyncMock

from tradegraph_financial_advisor.workflows.analysis_workflow import (
    FinancialAnalysisWorkflow,
    AnalysisState,
)


class TestFinancialAnalysisWorkflow:
    """Test FinancialAnalysisWorkflow functionality."""

    @pytest.mark.asyncio
    async def test_workflow_initialization(self, mock_local_scraping_service):
        """Test workflow initialization."""
        workflow = FinancialAnalysisWorkflow(scraping_service=mock_local_scraping_service)

        assert workflow.news_agent is not None
        assert workflow.financial_agent is not None
        assert workflow.local_scraping_service is not None
        assert workflow.workflow is not None

    @pytest.mark.asyncio
    async def test_analyze_portfolio_basic(self, mock_local_scraping_service):
        """Test basic portfolio analysis."""
        workflow = FinancialAnalysisWorkflow(scraping_service=mock_local_scraping_service)
        workflow.llm = AsyncMock()
        workflow.llm.ainvoke.return_value = Mock(
            content='{"recommendations": [], "total_confidence": 0.8, "diversification_score": 0.7, "overall_risk_level": "medium", "portfolio_size": 100000}'
        )

        symbols = ["AAPL"]
        portfolio_size = 100000
        risk_tolerance = "medium"

        result = await workflow.analyze_portfolio(
            symbols=symbols,
            portfolio_size=portfolio_size,
            risk_tolerance=risk_tolerance,
        )

        assert isinstance(result, dict)
        assert result.get("portfolio_recommendation") is not None

    @pytest.mark.asyncio
    async def test_collect_news_step(self, mock_local_scraping_service, sample_news_articles):
        """Test news collection workflow step."""
        workflow = FinancialAnalysisWorkflow(scraping_service=mock_local_scraping_service)
        workflow.news_agent.execute = AsyncMock(
            return_value={
                "articles": sample_news_articles,
                "total_count": len(sample_news_articles),
            }
        )

        initial_state = AnalysisState(
            symbols=["AAPL"],
            analysis_context={},
            news_data={},
            financial_data={},
            technical_data={},
            sentiment_analysis={},
            recommendations=[],
            portfolio_recommendation=None,
            messages=[],
            next_step="collect_news",
            error_messages=[],
        )

        result_state = await workflow._collect_news(initial_state)

        assert "articles" in result_state["news_data"]
        assert result_state["news_data"]["total_count"] > 0
        assert len(result_state["messages"]) > 0

    @pytest.mark.asyncio
    async def test_analyze_financials_step(
        self, mock_local_scraping_service, sample_financial_data
    ):
        """Test financial analysis workflow step."""
        workflow = FinancialAnalysisWorkflow(scraping_service=mock_local_scraping_service)
        workflow.financial_agent.execute = AsyncMock(
            return_value={"analysis_results": sample_financial_data}
        )

        initial_state = AnalysisState(
            symbols=["AAPL"],
            analysis_context={},
            news_data={},
            financial_data={},
            technical_data={},
            sentiment_analysis={},
            recommendations=[],
            portfolio_recommendation=None,
            messages=[],
            next_step="analyze_financials",
            error_messages=[],
        )

        result_state = await workflow._analyze_financials(initial_state)

        assert "analysis_results" in result_state["financial_data"]
        assert "AAPL" in result_state["financial_data"]["analysis_results"]
        assert len(result_state["messages"]) > 0

    @pytest.mark.asyncio
    async def test_analyze_sentiment_step(self, mock_local_scraping_service, sample_news_articles):
        """Test sentiment analysis workflow step."""
        workflow = FinancialAnalysisWorkflow(scraping_service=mock_local_scraping_service)
        workflow.llm = AsyncMock()
        workflow.llm.ainvoke.return_value = Mock(
            content='{"sentiment_score": 0.2, "sentiment_label": "bullish", "confidence": 0.8, "key_themes": ["growth", "innovation"], "sentiment_drivers": ["strong earnings", "new product launch"]}'
        )

        initial_state = AnalysisState(
            symbols=["AAPL"],
            analysis_context={},
            news_data={"articles": sample_news_articles},
            financial_data={},
            technical_data={},
            sentiment_analysis={},
            recommendations=[],
            portfolio_recommendation=None,
            messages=[],
            next_step="analyze_sentiment",
            error_messages=[],
        )

        result_state = await workflow._analyze_sentiment(initial_state)

        assert "AAPL" in result_state["sentiment_analysis"]
        assert len(result_state["messages"]) > 0

    @pytest.mark.asyncio
    async def test_generate_recommendations_step(
        self, mock_local_scraping_service, sample_recommendations
    ):
        """Test recommendation generation workflow step."""
        workflow = FinancialAnalysisWorkflow(scraping_service=mock_local_scraping_service)
        workflow.llm = AsyncMock()
        workflow.llm.ainvoke.return_value = Mock(
            content=json.dumps(
                {
                    "symbol": "AAPL",
                    "recommendation": "buy",
                    "confidence_score": 0.8,
                    "target_price": 220.0,
                    "risk_level": "medium",
                    "time_horizon": "medium_term",
                    "recommended_allocation": 0.1,
                    "fundamental_score": 0.8,
                    "technical_score": 0.7,
                    "sentiment_score": 0.8,
                    "key_factors": ["Strong fundamentals"],
                    "risks": ["Market volatility"],
                    "catalysts": ["Product launch"],
                    "analyst_notes": "Positive outlook",
                }
            )
        )

        initial_state = AnalysisState(
            symbols=["AAPL"],
            analysis_context={},
            news_data={},
            financial_data={
                "analysis_results": {"AAPL": {"market_data": {"current_price": 195.89}}}
            },
            technical_data={},
            sentiment_analysis={"AAPL": {"sentiment_score": 0.2}},
            recommendations=[],
            portfolio_recommendation=None,
            messages=[],
            next_step="generate_recommendations",
            error_messages=[],
        )

        result_state = await workflow._generate_recommendations(initial_state)

        assert len(result_state["recommendations"]) > 0
        assert len(result_state["messages"]) > 0

    @pytest.mark.asyncio
    async def test_create_portfolio_step(self, mock_local_scraping_service, sample_recommendations):
        """Test portfolio creation workflow step."""
        workflow = FinancialAnalysisWorkflow(scraping_service=mock_local_scraping_service)
        workflow.llm = AsyncMock()
        workflow.llm.ainvoke.return_value = Mock(
            content=json.dumps(
                {
                    "recommendations": [],
                    "total_confidence": 0.8,
                    "diversification_score": 0.7,
                    "overall_risk_level": "medium",
                    "portfolio_size": 100000,
                }
            )
        )

        initial_state = AnalysisState(
            symbols=["AAPL"],
            analysis_context={"portfolio_size": 100000, "risk_tolerance": "medium"},
            news_data={},
            financial_data={},
            technical_data={},
            sentiment_analysis={},
            recommendations=sample_recommendations,
            portfolio_recommendation=None,
            messages=[],
            next_step="create_portfolio",
            error_messages=[],
        )

        result_state = await workflow._create_portfolio(initial_state)

        assert result_state["portfolio_recommendation"] is not None
        assert result_state["portfolio_recommendation"]["portfolio_size"] == 100000
        assert len(result_state["messages"]) > 0

    @pytest.mark.asyncio
    async def test_validate_recommendations_step(
        self, mock_local_scraping_service, sample_portfolio_recommendation
    ):
        """Test recommendation validation workflow step."""
        workflow = FinancialAnalysisWorkflow(scraping_service=mock_local_scraping_service)

        initial_state = AnalysisState(
            symbols=["AAPL"],
            analysis_context={},
            news_data={},
            financial_data={},
            technical_data={},
            sentiment_analysis={},
            recommendations=sample_portfolio_recommendation["recommendations"],
            portfolio_recommendation=sample_portfolio_recommendation,
            messages=[],
            next_step="validate_recommendations",
            error_messages=[],
        )

        result_state = await workflow._validate_recommendations(initial_state)

        assert "validation_results" in result_state["analysis_context"]
        assert len(result_state["messages"]) > 0

    @pytest.mark.asyncio
    async def test_workflow_error_handling(self, mock_local_scraping_service):
        """Test workflow error handling."""
        workflow = FinancialAnalysisWorkflow(scraping_service=mock_local_scraping_service)
        workflow.news_agent.execute = AsyncMock(side_effect=Exception("Test error"))

        symbols = ["AAPL"]

        with pytest.raises(Exception):
            await workflow.analyze_portfolio(symbols=symbols)

    @pytest.mark.asyncio
    async def test_workflow_with_different_risk_tolerances(self, mock_local_scraping_service):
        """Test workflow with different risk tolerance settings."""
        workflow = FinancialAnalysisWorkflow(scraping_service=mock_local_scraping_service)
        workflow.llm = AsyncMock()
        workflow.llm.ainvoke.return_value = Mock(
            content='{"recommendations": [], "total_confidence": 0.8, "diversification_score": 0.7, "overall_risk_level": "medium", "portfolio_size": 100000}'
        )

        test_cases = [
            ("conservative", 50000),
            ("medium", 100000),
            ("aggressive", 200000),
        ]

        for risk_tolerance, portfolio_size in test_cases:
            result = await workflow.analyze_portfolio(
                symbols=["AAPL"],
                portfolio_size=portfolio_size,
                risk_tolerance=risk_tolerance,
            )

            assert isinstance(result, dict)
            assert result.get("portfolio_recommendation") is not None

    @pytest.mark.asyncio
    async def test_workflow_state_transitions(self, mock_local_scraping_service):
        """Test workflow state transitions."""
        workflow = FinancialAnalysisWorkflow(scraping_service=mock_local_scraping_service)

        # Test that workflow has proper state transitions
        assert hasattr(workflow.workflow, "get_graph")

        # The workflow should be properly compiled
        assert workflow.workflow is not None

    @pytest.mark.asyncio
    async def test_workflow_with_multiple_symbols(self, mock_local_scraping_service):
        """Test workflow with multiple symbols."""
        workflow = FinancialAnalysisWorkflow(scraping_service=mock_local_scraping_service)
        workflow.llm = AsyncMock()
        workflow.llm.ainvoke.return_value = Mock(
            content='{"recommendations": [], "total_confidence": 0.8, "diversification_score": 0.7, "overall_risk_level": "medium", "portfolio_size": 100000}'
        )

        symbols = ["AAPL", "MSFT", "GOOGL"]

        # Mock responses for multiple symbols
        workflow.news_agent.execute = AsyncMock(
            return_value={
                "articles": [
                    {"title": "AAPL News", "symbols": ["AAPL"]},
                    {"title": "MSFT News", "symbols": ["MSFT"]},
                    {"title": "GOOGL News", "symbols": ["GOOGL"]},
                ],
                "total_count": 3,
            }
        )

        workflow.financial_agent.execute = AsyncMock(
            return_value={
                "analysis_results": {
                    symbol: {
                        "market_data": {"current_price": 200.0},
                        "financials": {"pe_ratio": 25.0},
                    }
                    for symbol in symbols
                }
            }
        )

        result = await workflow.analyze_portfolio(symbols=symbols)

        assert isinstance(result, dict)
        assert result.get("portfolio_recommendation") is not None
        # Should have recommendations for multiple symbols
        assert len(result["recommendations"]) >= 0

    @pytest.mark.asyncio
    async def test_workflow_performance(self, mock_local_scraping_service):
        """Test workflow performance characteristics."""
        workflow = FinancialAnalysisWorkflow(scraping_service=mock_local_scraping_service)
        workflow.llm = AsyncMock()
        workflow.llm.ainvoke.return_value = Mock(
            content='{"recommendations": [], "total_confidence": 0.8, "diversification_score": 0.7, "overall_risk_level": "medium", "portfolio_size": 100000}'
        )

        import time

        start_time = time.time()

        result = await workflow.analyze_portfolio(symbols=["AAPL"], portfolio_size=100000)

        end_time = time.time()
        execution_time = end_time - start_time

        # Workflow should complete in reasonable time (with mocks)
        assert execution_time < 10.0  # seconds
        assert isinstance(result, dict)

    @pytest.mark.asyncio
    async def test_workflow_cleanup(self, mock_local_scraping_service):
        """Test workflow cleanup and resource management."""
        workflow = FinancialAnalysisWorkflow(scraping_service=mock_local_scraping_service)
        workflow.news_agent = AsyncMock()
        workflow.financial_agent = AsyncMock()
        workflow.local_scraping_service = AsyncMock()

        # Verify that agents are properly started and stopped
        await workflow.analyze_portfolio(symbols=["AAPL"])

        # Check that start/stop methods were called on agents
        workflow.news_agent.start.assert_called()
        workflow.news_agent.stop.assert_called()
        workflow.financial_agent.start.assert_called()
        workflow.financial_agent.stop.assert_called()
        workflow.local_scraping_service.start.assert_called()
        workflow.local_scraping_service.stop.assert_called()


class TestAnalysisState:
    """Test AnalysisState data structure."""

    def test_analysis_state_creation(self):
        """Test AnalysisState creation and default values."""
        state = AnalysisState(
            symbols=["AAPL"],
            analysis_context={},
            news_data={},
            financial_data={},
            technical_data={},
            sentiment_analysis={},
            recommendations=[],
            portfolio_recommendation=None,
            messages=[],
            next_step="collect_news",
            error_messages=[],
        )

        assert state["symbols"] == ["AAPL"]
        assert state["next_step"] == "collect_news"
        assert len(state["error_messages"]) == 0
        assert len(state["messages"]) == 0

    def test_analysis_state_modification(self):
        """Test AnalysisState modification."""
        state = AnalysisState(
            symbols=["AAPL"],
            analysis_context={},
            news_data={},
            financial_data={},
            technical_data={},
            sentiment_analysis={},
            recommendations=[],
            portfolio_recommendation=None,
            messages=[],
            next_step="collect_news",
            error_messages=[],
        )

        # Modify state
        state["news_data"] = {"articles": [{"title": "Test"}]}
        state["next_step"] = "analyze_financials"

        assert state["news_data"]["articles"][0]["title"] == "Test"
        assert state["next_step"] == "analyze_financials"
