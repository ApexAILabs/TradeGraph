import pytest
from datetime import datetime
from pydantic import ValidationError

from radgegraph_financial_advisor.models.financial_data import (
    NewsArticle,
    SentimentType,
    CompanyFinancials,
    MarketData,
    TechnicalIndicators,
    AnalysisContext,
)
from radgegraph_financial_advisor.models.recommendations import (
    TradingRecommendation,
    PortfolioRecommendation,
    RecommendationType,
    RiskLevel,
    TimeHorizon,
    AlertRecommendation,
)


class TestFinancialDataModels:
    """Test financial data models."""

    def test_news_article_creation(self):
        """Test NewsArticle model creation and validation."""
        article = NewsArticle(
            title="Test Article",
            url="https://example.com/test",
            content="Test content",
            source="test-source",
            published_at=datetime.now(),
            symbols=["AAPL", "MSFT"],
            sentiment=SentimentType.BULLISH,
            impact_score=0.8,
        )

        assert article.title == "Test Article"
        assert article.symbols == ["AAPL", "MSFT"]
        assert article.sentiment == SentimentType.BULLISH
        assert article.impact_score == 0.8

    def test_news_article_validation(self):
        """Test NewsArticle validation rules."""
        # Test invalid impact score
        with pytest.raises(ValidationError):
            NewsArticle(
                title="Test",
                url="https://example.com",
                content="Content",
                source="source",
                published_at=datetime.now(),
                impact_score=1.5,  # Invalid - should be <= 1.0
            )

        # Test negative impact score
        with pytest.raises(ValidationError):
            NewsArticle(
                title="Test",
                url="https://example.com",
                content="Content",
                source="source",
                published_at=datetime.now(),
                impact_score=-0.1,  # Invalid - should be >= 0.0
            )

    def test_sentiment_type_enum(self):
        """Test SentimentType enum values."""
        assert SentimentType.BULLISH == "bullish"
        assert SentimentType.BEARISH == "bearish"
        assert SentimentType.NEUTRAL == "neutral"

        # Test enum membership
        assert "bullish" in [item.value for item in SentimentType]
        assert "invalid_sentiment" not in [item.value for item in SentimentType]

    def test_company_financials_creation(self):
        """Test CompanyFinancials model creation."""
        financials = CompanyFinancials(
            symbol="AAPL",
            company_name="Apple Inc.",
            market_cap=3000000000000,
            pe_ratio=28.5,
            eps=6.88,
            revenue=394328000000,
            net_income=99803000000,
            debt_to_equity=1.73,
            current_ratio=1.05,
            return_on_equity=0.175,
            return_on_assets=0.225,
            price_to_book=5.02,
            dividend_yield=0.0047,
            beta=1.29,
            fifty_two_week_high=199.62,
            fifty_two_week_low=164.08,
            current_price=195.89,
        )

        assert financials.symbol == "AAPL"
        assert financials.company_name == "Apple Inc."
        assert financials.market_cap == 3000000000000
        assert financials.pe_ratio == 28.5

    def test_market_data_creation(self):
        """Test MarketData model creation."""
        market_data = MarketData(
            symbol="AAPL",
            current_price=195.89,
            change=2.15,
            change_percent=1.11,
            volume=45234567,
            market_cap=3000000000000,
            pe_ratio=28.5,
            timestamp=datetime.now(),
        )

        assert market_data.symbol == "AAPL"
        assert market_data.current_price == 195.89
        assert market_data.change == 2.15
        assert market_data.volume == 45234567

    def test_technical_indicators_creation(self):
        """Test TechnicalIndicators model creation."""
        technical = TechnicalIndicators(
            symbol="AAPL",
            sma_20=185.50,
            sma_50=180.25,
            ema_12=190.15,
            ema_26=187.80,
            rsi=65.2,
            macd=2.35,
            macd_signal=1.85,
            bollinger_upper=198.50,
            bollinger_lower=175.25,
            support_level=185.00,
            resistance_level=200.00,
            timestamp=datetime.now(),
        )

        assert technical.symbol == "AAPL"
        assert technical.sma_20 == 185.50
        assert technical.rsi == 65.2
        assert technical.support_level == 185.00

    def test_analysis_context_creation(self):
        """Test AnalysisContext model creation."""
        context = AnalysisContext(
            symbol="AAPL",
            news_articles=[],
            financials=None,
            market_data=None,
            technical_indicators=None,
            peer_comparison={},
            sector_performance={},
        )

        assert context.symbol == "AAPL"
        assert context.news_articles == []
        assert isinstance(context.analysis_timestamp, datetime)


class TestRecommendationModels:
    """Test recommendation models."""

    def test_recommendation_type_enum(self):
        """Test RecommendationType enum values."""
        assert RecommendationType.BUY == "buy"
        assert RecommendationType.SELL == "sell"
        assert RecommendationType.HOLD == "hold"
        assert RecommendationType.STRONG_BUY == "strong_buy"
        assert RecommendationType.STRONG_SELL == "strong_sell"

    def test_risk_level_enum(self):
        """Test RiskLevel enum values."""
        assert RiskLevel.LOW == "low"
        assert RiskLevel.MEDIUM == "medium"
        assert RiskLevel.HIGH == "high"
        assert RiskLevel.VERY_HIGH == "very_high"

    def test_time_horizon_enum(self):
        """Test TimeHorizon enum values."""
        assert TimeHorizon.SHORT_TERM == "short_term"
        assert TimeHorizon.MEDIUM_TERM == "medium_term"
        assert TimeHorizon.LONG_TERM == "long_term"

    def test_trading_recommendation_creation(self):
        """Test TradingRecommendation model creation."""
        recommendation = TradingRecommendation(
            symbol="AAPL",
            company_name="Apple Inc.",
            recommendation=RecommendationType.BUY,
            confidence_score=0.785,
            target_price=225.00,
            stop_loss=175.00,
            current_price=195.89,
            risk_level=RiskLevel.MEDIUM,
            time_horizon=TimeHorizon.MEDIUM_TERM,
            recommended_allocation=0.085,
            fundamental_score=0.80,
            technical_score=0.75,
            sentiment_score=0.80,
            key_factors=["Strong fundamentals", "Positive outlook"],
            risks=["Market volatility", "Competition"],
            catalysts=["Product launch", "Earnings beat"],
            analyst_notes="Strong buy recommendation",
            sector="Technology",
            expected_return=0.15,
        )

        assert recommendation.symbol == "AAPL"
        assert recommendation.recommendation == RecommendationType.BUY
        assert recommendation.confidence_score == 0.785
        assert recommendation.risk_level == RiskLevel.MEDIUM

    def test_trading_recommendation_validation(self):
        """Test TradingRecommendation validation rules."""
        # Test invalid confidence score (> 1.0)
        with pytest.raises(ValidationError):
            TradingRecommendation(
                symbol="AAPL",
                company_name="Apple Inc.",
                recommendation=RecommendationType.BUY,
                confidence_score=1.5,  # Invalid
                current_price=195.89,
                risk_level=RiskLevel.MEDIUM,
                time_horizon=TimeHorizon.MEDIUM_TERM,
                recommended_allocation=0.1,
                fundamental_score=0.8,
                technical_score=0.7,
                sentiment_score=0.8,
            )

        # Test invalid allocation (> 1.0)
        with pytest.raises(ValidationError):
            TradingRecommendation(
                symbol="AAPL",
                company_name="Apple Inc.",
                recommendation=RecommendationType.BUY,
                confidence_score=0.8,
                current_price=195.89,
                risk_level=RiskLevel.MEDIUM,
                time_horizon=TimeHorizon.MEDIUM_TERM,
                recommended_allocation=1.5,  # Invalid
                fundamental_score=0.8,
                technical_score=0.7,
                sentiment_score=0.8,
            )

    def test_portfolio_recommendation_creation(self):
        """Test PortfolioRecommendation model creation."""
        recommendations = [
            TradingRecommendation(
                symbol="AAPL",
                company_name="Apple Inc.",
                recommendation=RecommendationType.BUY,
                confidence_score=0.8,
                current_price=195.0,
                risk_level=RiskLevel.MEDIUM,
                time_horizon=TimeHorizon.MEDIUM_TERM,
                recommended_allocation=0.1,
                fundamental_score=0.8,
                technical_score=0.7,
                sentiment_score=0.8,
            )
        ]

        portfolio = PortfolioRecommendation(
            recommendations=recommendations,
            total_confidence=0.8,
            diversification_score=0.75,
            overall_risk_level=RiskLevel.MEDIUM,
            portfolio_size=100000,
            sector_weights={"Technology": 0.5},
            rebalancing_frequency="quarterly",
        )

        assert len(portfolio.recommendations) == 1
        assert portfolio.total_confidence == 0.8
        assert portfolio.overall_risk_level == RiskLevel.MEDIUM
        assert portfolio.portfolio_size == 100000

    def test_alert_recommendation_creation(self):
        """Test AlertRecommendation model creation."""
        alert = AlertRecommendation(
            symbol="AAPL",
            alert_type="price_target",
            message="AAPL approaching target price",
            urgency="high",
            trigger_conditions={"current_price": 195.0, "target_price": 200.0},
        )

        assert alert.symbol == "AAPL"
        assert alert.alert_type == "price_target"
        assert alert.urgency == "high"
        assert "current_price" in alert.trigger_conditions

    def test_model_serialization(self):
        """Test model serialization to dict."""
        recommendation = TradingRecommendation(
            symbol="AAPL",
            company_name="Apple Inc.",
            recommendation=RecommendationType.BUY,
            confidence_score=0.8,
            current_price=195.0,
            risk_level=RiskLevel.MEDIUM,
            time_horizon=TimeHorizon.MEDIUM_TERM,
            recommended_allocation=0.1,
            fundamental_score=0.8,
            technical_score=0.7,
            sentiment_score=0.8,
        )

        # Test dict conversion
        rec_dict = recommendation.model_dump()
        assert rec_dict["symbol"] == "AAPL"
        assert rec_dict["recommendation"] == "buy"
        assert rec_dict["risk_level"] == "medium"

        # Test JSON serialization
        import json

        json_str = recommendation.model_dump_json()
        parsed = json.loads(json_str)
        assert parsed["symbol"] == "AAPL"

    def test_model_deserialization(self):
        """Test model creation from dict."""
        data = {
            "symbol": "AAPL",
            "company_name": "Apple Inc.",
            "recommendation": "buy",
            "confidence_score": 0.8,
            "current_price": 195.0,
            "risk_level": "medium",
            "time_horizon": "medium_term",
            "recommended_allocation": 0.1,
            "fundamental_score": 0.8,
            "technical_score": 0.7,
            "sentiment_score": 0.8,
        }

        recommendation = TradingRecommendation(**data)
        assert recommendation.symbol == "AAPL"
        assert recommendation.recommendation == RecommendationType.BUY
        assert recommendation.risk_level == RiskLevel.MEDIUM

    def test_default_values(self):
        """Test model default values."""
        # Test NewsArticle defaults
        article = NewsArticle(
            title="Test",
            url="https://example.com",
            content="Content",
            source="source",
            published_at=datetime.now(),
        )
        assert article.symbols == []
        assert article.sentiment is None
        assert article.impact_score is None

        # Test TradingRecommendation defaults
        recommendation = TradingRecommendation(
            symbol="AAPL",
            company_name="Apple Inc.",
            recommendation=RecommendationType.BUY,
            confidence_score=0.8,
            current_price=195.0,
            risk_level=RiskLevel.MEDIUM,
            time_horizon=TimeHorizon.MEDIUM_TERM,
            recommended_allocation=0.1,
            fundamental_score=0.8,
            technical_score=0.7,
            sentiment_score=0.8,
        )
        assert recommendation.key_factors == []
        assert recommendation.risks == []
        assert recommendation.catalysts == []
        assert isinstance(recommendation.analysis_timestamp, datetime)

    def test_optional_fields(self):
        """Test models with optional fields."""
        # CompanyFinancials with minimal data
        financials = CompanyFinancials(symbol="AAPL", company_name="Apple Inc.")
        assert financials.symbol == "AAPL"
        assert financials.market_cap is None
        assert financials.pe_ratio is None

        # TradingRecommendation with minimal data
        recommendation = TradingRecommendation(
            symbol="AAPL",
            company_name="Apple Inc.",
            recommendation=RecommendationType.HOLD,
            confidence_score=0.5,
            current_price=195.0,
            risk_level=RiskLevel.MEDIUM,
            time_horizon=TimeHorizon.MEDIUM_TERM,
            recommended_allocation=0.05,
            fundamental_score=0.5,
            technical_score=0.5,
            sentiment_score=0.5,
        )
        assert recommendation.target_price is None
        assert recommendation.stop_loss is None
        assert recommendation.expected_return is None
