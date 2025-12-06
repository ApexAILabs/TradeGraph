import asyncio
import json
from typing import Any, Dict, List, Optional, TypedDict, Annotated
from datetime import datetime
from langchain_core.messages import HumanMessage, AIMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from loguru import logger

from ..agents.news_agent import NewsReaderAgent
from ..agents.financial_agent import FinancialAnalysisAgent
from ..agents.recommendation_engine import TradingRecommendationEngine
from ..services.firecrawl_service import FirecrawlService
from ..models.financial_data import AnalysisContext
from ..models.recommendations import (
    TradingRecommendation,
    PortfolioRecommendation,
    RecommendationType,
    RiskLevel,
    TimeHorizon
)
from ..config.settings import settings


class AnalysisState(TypedDict):
    symbols: List[str]
    analysis_context: Dict[str, Any]
    news_data: Dict[str, Any]
    financial_data: Dict[str, Any]
    technical_data: Dict[str, Any]
    sentiment_analysis: Dict[str, Any]
    recommendations: List[Dict[str, Any]]
    portfolio_recommendation: Optional[Dict[str, Any]]
    messages: List[Any]
    next_step: str
    error_messages: List[str]


class FinancialAnalysisWorkflow:
    def __init__(self):
        self.llm = ChatOpenAI(
            model="gpt-4",
            temperature=0.1,
            api_key=settings.openai_api_key
        )
        self.news_agent = NewsReaderAgent()
        self.financial_agent = FinancialAnalysisAgent()
        self.recommendation_engine = TradingRecommendationEngine()
        self.firecrawl_service = FirecrawlService()
        self.workflow = None
        self._build_workflow()

    def _build_workflow(self) -> None:
        workflow = StateGraph(AnalysisState)

        # Add nodes
        workflow.add_node("collect_news", self._collect_news)
        workflow.add_node("analyze_financials", self._analyze_financials)
        workflow.add_node("analyze_sentiment", self._analyze_sentiment)
        workflow.add_node("generate_recommendations", self._generate_recommendations)
        workflow.add_node("create_portfolio", self._create_portfolio)
        workflow.add_node("validate_recommendations", self._validate_recommendations)

        # Add edges
        workflow.set_entry_point("collect_news")
        workflow.add_edge("collect_news", "analyze_financials")
        workflow.add_edge("analyze_financials", "analyze_sentiment")
        workflow.add_edge("analyze_sentiment", "generate_recommendations")
        workflow.add_edge("generate_recommendations", "create_portfolio")
        workflow.add_edge("create_portfolio", "validate_recommendations")
        workflow.add_edge("validate_recommendations", END)

        self.workflow = workflow.compile()

    async def analyze_portfolio(
        self,
        symbols: List[str],
        portfolio_size: float = None,
        risk_tolerance: str = "medium",
        time_horizon: str = "medium_term"
    ) -> Dict[str, Any]:

        if portfolio_size is None:
            portfolio_size = settings.default_portfolio_size

        initial_state = AnalysisState(
            symbols=symbols,
            analysis_context={
                "portfolio_size": portfolio_size,
                "risk_tolerance": risk_tolerance,
                "time_horizon": time_horizon,
                "analysis_timestamp": datetime.now().isoformat()
            },
            news_data={},
            financial_data={},
            technical_data={},
            sentiment_analysis={},
            recommendations=[],
            portfolio_recommendation=None,
            messages=[],
            next_step="collect_news",
            error_messages=[]
        )

        try:
            # Start agents
            await self.news_agent.start()
            await self.financial_agent.start()
            await self.firecrawl_service.start()

            # Execute workflow
            result = await self.workflow.ainvoke(initial_state)

            # Return the complete analysis state including sentiment analysis
            analysis_result = {
                "portfolio_recommendation": result.get("portfolio_recommendation"),
                "sentiment_analysis": result.get("sentiment_analysis", {}),
                "news_data": result.get("news_data", {}),
                "financial_data": result.get("financial_data", {}),
                "recommendations": result.get("recommendations", []),
                "analysis_context": result.get("analysis_context", {})
            }

            return analysis_result

        except Exception as e:
            logger.error(f"Workflow execution failed: {str(e)}")
            raise
        finally:
            # Cleanup
            await self.news_agent.stop()
            await self.financial_agent.stop()
            await self.firecrawl_service.stop()

    async def _collect_news(self, state: AnalysisState) -> AnalysisState:
        try:
            logger.info(f"Collecting news for symbols: {state['symbols']}")

            # Collect news from multiple sources
            news_input = {
                "symbols": state["symbols"],
                "timeframe_hours": 48,
                "max_articles": 100
            }

            news_result = await self.news_agent.execute(news_input)

            # Also use Firecrawl for additional news scraping
            firecrawl_articles = await self.firecrawl_service.scrape_news_websites(
                state["symbols"],
                max_articles_per_source=20
            )

            # Combine results
            all_articles = news_result.get("articles", [])
            all_articles.extend([article.dict() for article in firecrawl_articles])

            state["news_data"] = {
                "articles": all_articles,
                "total_count": len(all_articles),
                "collection_timestamp": datetime.now().isoformat()
            }

            state["messages"].append(
                AIMessage(content=f"Collected {len(all_articles)} news articles")
            )

        except Exception as e:
            error_msg = f"News collection failed: {str(e)}"
            logger.error(error_msg)
            state["error_messages"].append(error_msg)

        return state

    async def _analyze_financials(self, state: AnalysisState) -> AnalysisState:
        try:
            logger.info(f"Analyzing financials for symbols: {state['symbols']}")

            financial_input = {
                "symbols": state["symbols"],
                "include_financials": True,
                "include_technical": True,
                "include_market_data": True
            }

            financial_result = await self.financial_agent.execute(financial_input)

            state["financial_data"] = financial_result

            # Scrape SEC filings for additional fundamental analysis
            for symbol in state["symbols"]:
                try:
                    filings = await self.firecrawl_service.scrape_financial_reports(symbol)
                    if filings:
                        if "sec_filings" not in state["financial_data"]:
                            state["financial_data"]["sec_filings"] = {}
                        state["financial_data"]["sec_filings"][symbol] = filings
                except Exception as e:
                    logger.warning(f"Failed to scrape SEC filings for {symbol}: {str(e)}")

            state["messages"].append(
                AIMessage(content="Completed financial analysis")
            )

        except Exception as e:
            error_msg = f"Financial analysis failed: {str(e)}"
            logger.error(error_msg)
            state["error_messages"].append(error_msg)

        return state

    async def _analyze_sentiment(self, state: AnalysisState) -> AnalysisState:
        try:
            logger.info("Analyzing sentiment from news articles")

            articles = state["news_data"].get("articles", [])

            sentiment_analysis = {}

            for symbol in state["symbols"]:
                symbol_articles = [
                    article for article in articles
                    if symbol in getattr(article, 'symbols', [])
                ]

                if not symbol_articles:
                    continue

                # Use LLM for advanced sentiment analysis with news summary
                articles_text = chr(10).join([f"- {article.title}: {article.content[:300]}..." for article in symbol_articles[:10]])

                sentiment_prompt = f"""
                Analyze the sentiment of the following news articles about {symbol}.
                Provide a sentiment score from -1 (very bearish) to 1 (very bullish),
                identify key themes, sentiment drivers, and create a concise summary.

                Articles:
                {articles_text}

                Respond with a JSON object containing:
                - sentiment_score: float between -1 and 1
                - sentiment_label: "bullish", "bearish", "neutral", or "warning"
                - confidence: float between 0 and 1
                - key_themes: list of strings
                - sentiment_drivers: list of strings
                - news_summary: string (2-3 sentences summarizing the key news points)
                - article_count: number of articles analyzed
                - articles: list of objects with title, url, sentiment_contribution (positive/negative/neutral)
                """

                response = await self.llm.ainvoke([HumanMessage(content=sentiment_prompt)])

                try:
                    import json
                    sentiment_data = json.loads(response.content)
                    sentiment_analysis[symbol] = sentiment_data
                except:
                    # Fallback simple sentiment with news data
                    article_summaries = [{"title": getattr(article, 'title', ''), "url": getattr(article, 'url', ''), "sentiment_contribution": "neutral"} for article in symbol_articles[:5]]

                    sentiment_analysis[symbol] = {
                        "sentiment_score": 0.0,
                        "sentiment_label": "neutral",
                        "confidence": 0.5,
                        "key_themes": [],
                        "sentiment_drivers": [],
                        "news_summary": f"Analysis of {len(symbol_articles)} news articles about {symbol}. Detailed AI analysis temporarily unavailable.",
                        "article_count": len(symbol_articles),
                        "articles": article_summaries
                    }

            state["sentiment_analysis"] = sentiment_analysis

            state["messages"].append(
                AIMessage(content="Completed sentiment analysis")
            )

        except Exception as e:
            error_msg = f"Sentiment analysis failed: {str(e)}"
            logger.error(error_msg)
            state["error_messages"].append(error_msg)

        return state

    async def _generate_recommendations(self, state: AnalysisState) -> AnalysisState:
        try:
            logger.info("Generating trading recommendations")

            recommendations = []

            for symbol in state["symbols"]:
                try:
                    # Get data for this symbol
                    financial_data = state["financial_data"].get("analysis_results", {}).get(symbol, {})
                    sentiment_data = state["sentiment_analysis"].get(symbol, {})

                    # Generate recommendation using LLM
                    recommendation_prompt = f"""
                    Based on the following financial and sentiment analysis for {symbol},
                    generate a trading recommendation.

                    Financial Data:
                    {str(financial_data)}

                    Sentiment Analysis:
                    {str(sentiment_data)}

                    Provide a recommendation with the following JSON structure:
                    {{
                        "symbol": "{symbol}",
                        "recommendation": "buy|sell|hold|strong_buy|strong_sell",
                        "confidence_score": 0.0-1.0,
                        "target_price": number or null,
                        "stop_loss": number or null,
                        "risk_level": "low|medium|high|very_high",
                        "time_horizon": "short_term|medium_term|long_term",
                        "recommended_allocation": 0.0-1.0,
                        "fundamental_score": 0.0-1.0,
                        "technical_score": 0.0-1.0,
                        "sentiment_score": 0.0-1.0,
                        "key_factors": ["factor1", "factor2"],
                        "risks": ["risk1", "risk2"],
                        "catalysts": ["catalyst1", "catalyst2"],
                        "analyst_notes": "detailed analysis"
                    }}
                    """

                    response = await self.llm.ainvoke([HumanMessage(content=recommendation_prompt)])

                    try:
                        import json
                        rec_data = json.loads(response.content)

                        # Add required fields
                        current_price = financial_data.get("market_data", {}).get("current_price", 100.0)
                        rec_data["current_price"] = current_price
                        rec_data["company_name"] = symbol  # Fallback

                        # Handle missing sentiment_score
                        if rec_data.get("sentiment_score") is None:
                            rec_data["sentiment_score"] = 0.5

                        # Calculate recommended allocation using new logic
                        confidence_score = rec_data.get("confidence_score", 0.5)
                        risk_level = RiskLevel(rec_data.get("risk_level", "medium"))
                        recommendation_type = RecommendationType(rec_data.get("recommendation", "hold"))
                        risk_preferences = {"risk_tolerance": state["analysis_context"].get("risk_tolerance", "medium")}

                        # Use position sizing logic
                        rec_data["recommended_allocation"] = self.recommendation_engine._calculate_position_size(
                            confidence_score=confidence_score,
                            risk_level=risk_level,
                            risk_preferences=risk_preferences,
                            recommendation_type=recommendation_type
                        )

                        recommendation = TradingRecommendation(**rec_data)
                        recommendations.append(recommendation)

                    except Exception as e:
                        logger.warning(f"Failed to parse recommendation for {symbol}: {str(e)}")

                except Exception as e:
                    logger.warning(f"Failed to generate recommendation for {symbol}: {str(e)}")

            # Optimize portfolio allocations after all recommendations are collected
            if recommendations:
                portfolio_constraints = {
                    "portfolio_size": state["analysis_context"].get("portfolio_size", 100000),
                    "max_positions": 10
                }
                optimized_recs = await self.recommendation_engine._optimize_allocations(
                    recommendations,
                    portfolio_constraints
                )
                state["recommendations"] = [rec.dict() for rec in optimized_recs]
            else:
                state["recommendations"] = []

            state["messages"].append(
                AIMessage(content=f"Generated {len(recommendations)} trading recommendations")
            )

        except Exception as e:
            error_msg = f"Recommendation generation failed: {str(e)}"
            logger.error(error_msg)
            state["error_messages"].append(error_msg)

        return state

    async def _create_portfolio(self, state: AnalysisState) -> AnalysisState:
        try:
            logger.info("Creating portfolio recommendation")

            recommendations = state["recommendations"]
            portfolio_size = state["analysis_context"]["portfolio_size"]
            risk_tolerance = state["analysis_context"]["risk_tolerance"]

            if not recommendations:
                state["portfolio_recommendation"] = None
                return state

            # Create portfolio using LLM
            portfolio_prompt = f"""
            Create an optimal portfolio allocation based on the following individual recommendations
            and portfolio constraints:

            Individual Recommendations:
            {str(recommendations)}

            Portfolio Size: ${portfolio_size:,.2f}
            Risk Tolerance: {risk_tolerance}

            Generate a portfolio recommendation with this JSON structure:
            {{
                "recommendations": [...], // Include the individual recommendations
                "total_confidence": 0.0-1.0,
                "diversification_score": 0.0-1.0,
                "expected_return": percentage,
                "expected_volatility": percentage,
                "sector_weights": {{"sector": weight}},
                "overall_risk_level": "low|medium|high|very_high",
                "rebalancing_frequency": "monthly|quarterly|semi_annual",
                "portfolio_size": {portfolio_size}
            }}

            Ensure allocations sum to 100% and align with risk tolerance.
            """

            response = await self.llm.ainvoke([HumanMessage(content=portfolio_prompt)])

            try:
                import json
                portfolio_data = json.loads(response.content)
                portfolio_data["recommendations"] = recommendations  # Ensure recommendations are included
                state["portfolio_recommendation"] = portfolio_data

            except Exception as e:
                logger.warning(f"Failed to parse portfolio recommendation: {str(e)}")
                # Create basic portfolio
                # Map risk tolerance to valid enum values
                risk_level_mapping = {
                    "conservative": "low",
                    "moderate": "medium",
                    "aggressive": "high",
                    "very_aggressive": "very_high"
                }
                mapped_risk_level = risk_level_mapping.get(risk_tolerance, "medium")

                state["portfolio_recommendation"] = {
                    "recommendations": recommendations,
                    "total_confidence": 0.7,
                    "diversification_score": 0.6,
                    "overall_risk_level": mapped_risk_level,
                    "portfolio_size": portfolio_size
                }

            state["messages"].append(
                AIMessage(content="Created portfolio recommendation")
            )

        except Exception as e:
            error_msg = f"Portfolio creation failed: {str(e)}"
            logger.error(error_msg)
            state["error_messages"].append(error_msg)

        return state

    async def _validate_recommendations(self, state: AnalysisState) -> AnalysisState:
        try:
            logger.info("Validating recommendations")

            # Perform validation checks
            validation_results = []

            # Check allocation totals
            if state["portfolio_recommendation"]:
                total_allocation = sum(
                    rec.get("recommended_allocation", 0)
                    for rec in state["portfolio_recommendation"]["recommendations"]
                )

                if abs(total_allocation - 1.0) > 0.1:  # Allow 10% tolerance
                    validation_results.append(f"Warning: Total allocation is {total_allocation:.2%}")

            # Check for obvious conflicts
            buy_count = sum(
                1 for rec in state["recommendations"]
                if rec.get("recommendation") in ["buy", "strong_buy"]
            )

            sell_count = sum(
                1 for rec in state["recommendations"]
                if rec.get("recommendation") in ["sell", "strong_sell"]
            )

            if buy_count == 0 and sell_count == 0:
                validation_results.append("Warning: No buy or sell recommendations generated")

            state["analysis_context"]["validation_results"] = validation_results

            state["messages"].append(
                AIMessage(content=f"Validation completed with {len(validation_results)} notes")
            )

        except Exception as e:
            error_msg = f"Validation failed: {str(e)}"
            logger.error(error_msg)
            state["error_messages"].append(error_msg)

        return state