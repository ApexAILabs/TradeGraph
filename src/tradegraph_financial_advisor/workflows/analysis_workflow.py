import json
from typing import Any, Dict, List, Optional, TypedDict
from datetime import datetime
from langchain_core.messages import HumanMessage, AIMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from loguru import logger

from ..agents.news_agent import NewsReaderAgent
from ..agents.financial_agent import FinancialAnalysisAgent
from ..agents.recommendation_engine import TradingRecommendationEngine
from ..services.local_scraping_service import LocalScrapingService
from ..services.channel_stream_service import FinancialNewsChannelService
from ..models.recommendations import (
    TradingRecommendation,
    RecommendationType,
    RiskLevel,
    TimeHorizon,
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
    channel_streams: Dict[str, Any]


class FinancialAnalysisWorkflow:
    def __init__(
        self,
        scraping_service: Optional[LocalScrapingService] = None,
        llm_model_name: str = "gpt-5-nano",
    ):
        self.llm_model_name = llm_model_name
        self.llm = ChatOpenAI(
            model=self.llm_model_name, temperature=0.1, api_key=settings.openai_api_key
        )
        self.news_agent = NewsReaderAgent()
        self.financial_agent = FinancialAnalysisAgent()
        self.recommendation_engine = TradingRecommendationEngine(
            model_name=self.llm_model_name
        )
        self.local_scraping_service = scraping_service or LocalScrapingService()
        self.channel_service = FinancialNewsChannelService()
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
        time_horizon: str = "medium_term",
    ) -> Dict[str, Any]:

        if portfolio_size is None:
            portfolio_size = settings.default_portfolio_size

        initial_state = AnalysisState(
            symbols=symbols,
            analysis_context={
                "portfolio_size": portfolio_size,
                "risk_tolerance": risk_tolerance,
                "time_horizon": time_horizon,
                "analysis_timestamp": datetime.now().isoformat(),
            },
            news_data={},
            financial_data={},
            technical_data={},
            sentiment_analysis={},
            recommendations=[],
            portfolio_recommendation=None,
            messages=[],
            next_step="collect_news",
            error_messages=[],
            channel_streams={},
        )

        try:
            # Start agents
            await self.news_agent.start()
            await self.financial_agent.start()
            await self.local_scraping_service.start()

            # Execute workflow
            result = await self.workflow.ainvoke(initial_state)

            # Return the complete analysis state including sentiment analysis
            analysis_result = {
                "portfolio_recommendation": result.get("portfolio_recommendation"),
                "sentiment_analysis": result.get("sentiment_analysis", {}),
                "news_data": result.get("news_data", {}),
                "financial_data": result.get("financial_data", {}),
                "recommendations": result.get("recommendations", []),
                "analysis_context": result.get("analysis_context", {}),
                "channel_streams": result.get("channel_streams", {}),
            }

            return analysis_result

        except Exception as e:
            logger.error(f"Workflow execution failed: {str(e)}")
            raise
        finally:
            # Cleanup
            await self.news_agent.stop()
            await self.financial_agent.stop()
            await self.local_scraping_service.stop()
            await self.channel_service.close()

    async def _collect_news(self, state: AnalysisState) -> AnalysisState:
        try:
            logger.info(f"Collecting news for symbols: {state['symbols']}")

            # Collect news from multiple sources
            news_input = {
                "symbols": state["symbols"],
                "timeframe_hours": 48,
                "max_articles": 100,
            }

            news_result = await self.news_agent.execute(news_input)
            if not isinstance(news_result, dict):
                news_result = {}

            local_articles = await self.local_scraping_service.search_and_scrape_news(
                state["symbols"]
            )
            if not isinstance(local_articles, list):
                local_articles = []

            news_articles = news_result.get("articles", [])
            if not isinstance(news_articles, list):
                news_articles = []

            combined_news = [
                self._article_to_dict(article) for article in news_articles
            ]
            combined_news.extend(
                self._article_to_dict(article) for article in local_articles
            )

            state["news_data"] = {
                "articles": combined_news,
                "total_count": len(combined_news),
                "collection_timestamp": datetime.now().isoformat(),
            }

            # Capture websocket channel payloads for downstream reporting
            try:
                channel_payloads = await self.channel_service.collect_all_channels(
                    state["symbols"]
                )
                state["channel_streams"] = channel_payloads
            except Exception as channel_exc:
                logger.warning(
                    f"Failed to collect channel streams: {channel_exc}"
                )

            state["messages"].append(
                AIMessage(content=f"Collected {len(combined_news)} news articles")
            )

        except Exception as e:
            error_msg = f"News collection failed: {str(e)}"
            logger.error(error_msg)
            state["error_messages"].append(error_msg)
            raise

        return state

    async def _analyze_financials(self, state: AnalysisState) -> AnalysisState:
        try:
            logger.info(f"Analyzing financials for symbols: {state['symbols']}")

            financial_input = {
                "symbols": state["symbols"],
                "include_financials": True,
                "include_technical": True,
                "include_market_data": True,
            }

            financial_result = await self.financial_agent.execute(financial_input)

            state["financial_data"] = financial_result

            # Scrape SEC filings for additional fundamental analysis
            for symbol in state["symbols"]:
                try:
                    filings = await self.local_scraping_service.search_and_scrape_financial_reports(
                        symbol
                    )
                    if filings:
                        if "sec_filings" not in state["financial_data"]:
                            state["financial_data"]["sec_filings"] = {}
                        state["financial_data"]["sec_filings"][symbol] = filings
                except Exception as e:
                    logger.warning(
                        f"Failed to scrape SEC filings for {symbol}: {str(e)}"
                    )

            state["messages"].append(AIMessage(content="Completed financial analysis"))

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
                    article
                    for article in articles
                    if symbol in self._get_article_symbols(article)
                ]

                if not symbol_articles:
                    continue

                # Use LLM for advanced sentiment analysis with news summary
                articles_text = chr(10).join(
                    [
                        f"- {self._get_article_value(article, 'title', 'Article')}: "
                        f"{(self._get_article_value(article, 'content', '') or '')[:300]}..."
                        for article in symbol_articles[:10]
                    ]
                )

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

                response = await self.llm.ainvoke(
                    [HumanMessage(content=sentiment_prompt)]
                )

                try:
                    import json

                    sentiment_data = json.loads(response.content)
                    sentiment_analysis[symbol] = sentiment_data
                except Exception:
                    # Fallback simple sentiment with news data
                    article_summaries = [
                        {
                            "title": self._get_article_value(article, "title", ""),
                            "url": self._get_article_value(article, "url", ""),
                            "sentiment_contribution": "neutral",
                        }
                        for article in symbol_articles[:5]
                    ]

                    sentiment_analysis[symbol] = {
                        "sentiment_score": 0.0,
                        "sentiment_label": "neutral",
                        "confidence": 0.5,
                        "key_themes": [],
                        "sentiment_drivers": [],
                        "news_summary": f"Analysis of {len(symbol_articles)} news articles about {symbol}. Detailed AI analysis temporarily unavailable.",
                        "article_count": len(symbol_articles),
                        "articles": article_summaries,
                    }

            state["sentiment_analysis"] = sentiment_analysis

            state["messages"].append(AIMessage(content="Completed sentiment analysis"))

        except Exception as e:
            error_msg = f"Sentiment analysis failed: {str(e)}"
            logger.error(error_msg)
            state["error_messages"].append(error_msg)

        return state

    async def _generate_recommendations(self, state: AnalysisState) -> AnalysisState:
        try:
            logger.info("Generating trading recommendations")

            recommendations: List[TradingRecommendation] = []

            for symbol in state["symbols"]:
                try:
                    # Get analysis inputs for this symbol
                    financial_data = (
                        state["financial_data"]
                        .get("analysis_results", {})
                        .get(symbol, {})
                    )
                    sentiment_data = state["sentiment_analysis"].get(symbol, {})

                    # Generate recommendation using LLM for additional context
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

                    parsed_response: Optional[Dict[str, Any]] = None
                    try:
                        response = await self.llm.ainvoke(
                            [HumanMessage(content=recommendation_prompt)]
                        )
                        if response and getattr(response, "content", None):
                            potential_payload = json.loads(response.content)
                            if isinstance(potential_payload, dict):
                                parsed_response = potential_payload
                    except Exception as llm_error:
                        logger.warning(
                            f"Failed to obtain structured recommendation for {symbol}: {str(llm_error)}"
                        )

                    recommendation = self._build_recommendation_model(
                        symbol=symbol,
                        financial_data=financial_data,
                        sentiment_data=sentiment_data,
                        state=state,
                        model_response=parsed_response,
                    )
                    recommendations.append(recommendation)

                except Exception as e:
                    logger.warning(
                        f"Failed to generate recommendation for {symbol}: {str(e)}"
                    )

            # Optimize portfolio allocations after all recommendations are collected
            if recommendations:
                portfolio_constraints = {
                    "portfolio_size": state["analysis_context"].get(
                        "portfolio_size", 100000
                    ),
                    "max_positions": 10,
                }
                optimized_recs = await self.recommendation_engine._optimize_allocations(
                    recommendations,
                    portfolio_constraints,
                )
                state["recommendations"] = [rec.dict() for rec in optimized_recs]
            else:
                state["recommendations"] = []

            state["messages"].append(
                AIMessage(
                    content=f"Generated {len(recommendations)} trading recommendations"
                )
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
                portfolio_data["recommendations"] = (
                    recommendations  # Ensure recommendations are included
                )
                state["portfolio_recommendation"] = portfolio_data

            except Exception as e:
                logger.warning(f"Failed to parse portfolio recommendation: {str(e)}")
                # Create basic portfolio
                # Map risk tolerance to valid enum values
                risk_level_mapping = {
                    "conservative": "low",
                    "moderate": "medium",
                    "aggressive": "high",
                    "very_aggressive": "very_high",
                }
                mapped_risk_level = risk_level_mapping.get(risk_tolerance, "medium")

                state["portfolio_recommendation"] = {
                    "recommendations": recommendations,
                    "total_confidence": 0.7,
                    "diversification_score": 0.6,
                    "overall_risk_level": mapped_risk_level,
                    "portfolio_size": portfolio_size,
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
                    validation_results.append(
                        f"Warning: Total allocation is {total_allocation:.2%}"
                    )

            # Check for obvious conflicts
            buy_count = sum(
                1
                for rec in state["recommendations"]
                if rec.get("recommendation") in ["buy", "strong_buy"]
            )

            sell_count = sum(
                1
                for rec in state["recommendations"]
                if rec.get("recommendation") in ["sell", "strong_sell"]
            )

            if buy_count == 0 and sell_count == 0:
                validation_results.append(
                    "Warning: No buy or sell recommendations generated"
                )

            state["analysis_context"]["validation_results"] = validation_results

            state["messages"].append(
                AIMessage(
                    content=f"Validation completed with {len(validation_results)} notes"
                )
            )

        except Exception as e:
            error_msg = f"Validation failed: {str(e)}"
            logger.error(error_msg)
            state["error_messages"].append(error_msg)

        return state

    def _build_recommendation_model(
        self,
        symbol: str,
        financial_data: Dict[str, Any],
        sentiment_data: Dict[str, Any],
        state: AnalysisState,
        model_response: Optional[Dict[str, Any]] = None,
    ) -> TradingRecommendation:
        response_payload = model_response if isinstance(model_response, dict) else {}
        analysis_context = (
            state.get("analysis_context", {}) if isinstance(state, dict) else {}
        )

        market_data = financial_data.get("market_data") or {}
        fundamentals = financial_data.get("financials") or {}
        technicals = financial_data.get("technical_indicators") or {}

        portfolio_size = (
            analysis_context.get("portfolio_size") or settings.default_portfolio_size
        )
        current_price = self._extract_current_price(
            market_data, fundamentals, technicals
        )
        company_name = (
            response_payload.get("company_name")
            or fundamentals.get("company_name")
            or market_data.get("symbol")
            or symbol
        )

        raw_sentiment = response_payload.get("sentiment_score")
        if raw_sentiment is None:
            raw_sentiment = sentiment_data.get("sentiment_score")
        sentiment_score = self._normalize_sentiment_score(raw_sentiment)

        raw_confidence = response_payload.get("confidence_score")
        if raw_confidence is None:
            raw_confidence = sentiment_data.get("confidence")
        if raw_confidence is None:
            raw_confidence = sentiment_score
        confidence_score = float(
            min(0.95, max(0.3, raw_confidence if raw_confidence is not None else 0.5))
        )

        risk_level = self._normalize_risk_level(
            response_payload.get("risk_level")
            or analysis_context.get("risk_tolerance")
            or "medium"
        )
        time_horizon = self._normalize_time_horizon(
            response_payload.get("time_horizon")
            or analysis_context.get("time_horizon")
            or "medium_term"
        )
        recommendation_type = self._normalize_recommendation_type(
            response_payload.get("recommendation"), sentiment_score
        )

        fundamental_score = self._clamp_score(
            response_payload.get("fundamental_score"),
            self._compute_fundamental_score(fundamentals),
        )
        technical_score = self._clamp_score(
            response_payload.get("technical_score"),
            self._compute_technical_score(technicals),
        )

        sentiment_score = round(
            sentiment_score if sentiment_score is not None else 0.5, 3
        )
        key_factors = self._ensure_list(
            response_payload.get("key_factors")
            or sentiment_data.get("key_themes")
            or [f"Market outlook for {symbol}"]
        )
        risks = self._ensure_list(
            response_payload.get("risks")
            or sentiment_data.get("risks")
            or [f"Potential volatility in {symbol}"]
        )
        catalysts = self._ensure_list(
            response_payload.get("catalysts")
            or sentiment_data.get("sentiment_drivers")
            or []
        )

        analyst_notes = (
            response_payload.get("analyst_notes")
            or sentiment_data.get("news_summary")
            or f"Auto-generated analysis for {symbol}."
        )

        target_price = response_payload.get("target_price")
        if target_price in (None, 0):
            target_price = self._estimate_target_price(
                current_price, recommendation_type, sentiment_score
            )

        stop_loss = response_payload.get("stop_loss")
        if stop_loss in (None, 0):
            stop_loss = self._estimate_stop_loss(
                current_price, risk_level, recommendation_type
            )

        risk_preferences = {
            "risk_tolerance": analysis_context.get("risk_tolerance", "medium")
        }
        recommended_allocation = self.recommendation_engine._calculate_position_size(
            confidence_score=confidence_score,
            risk_level=risk_level,
            risk_preferences=risk_preferences,
            recommendation_type=recommendation_type,
        )
        recommended_allocation = float(min(1.0, max(0.01, recommended_allocation)))

        max_position_size = round(float(portfolio_size) * recommended_allocation, 2)
        expected_return = round(
            ((target_price - current_price) / current_price) if current_price else 0.0,
            3,
        )
        expected_volatility = self._map_expected_volatility(risk_level)
        sharpe_ratio = None
        if expected_volatility:
            sharpe_ratio = round((expected_return - 0.02) / expected_volatility, 2)

        market_cap_value = (
            response_payload.get("market_cap")
            or fundamentals.get("market_cap")
            or market_data.get("market_cap")
        )

        recommendation_data = {
            "symbol": symbol,
            "company_name": company_name,
            "recommendation": recommendation_type,
            "confidence_score": round(confidence_score, 3),
            "target_price": round(target_price, 2),
            "stop_loss": round(stop_loss, 2) if stop_loss is not None else None,
            "current_price": round(current_price, 2),
            "risk_level": risk_level,
            "time_horizon": time_horizon,
            "recommended_allocation": recommended_allocation,
            "max_position_size": max_position_size,
            "fundamental_score": round(fundamental_score, 3),
            "technical_score": round(technical_score, 3),
            "sentiment_score": sentiment_score,
            "key_factors": key_factors,
            "risks": risks,
            "catalysts": catalysts,
            "analyst_notes": analyst_notes,
            "sector": fundamentals.get("sector")
            or response_payload.get("sector")
            or "Unknown",
            "market_cap_category": self._categorize_market_cap(market_cap_value),
            "expected_return": expected_return,
            "expected_volatility": expected_volatility,
            "sharpe_ratio": sharpe_ratio,
        }

        return TradingRecommendation(**recommendation_data)

    def _normalize_sentiment_score(self, value: Optional[float]) -> float:
        if value is None:
            return 0.5
        try:
            numeric_value = float(value)
        except (TypeError, ValueError):
            return 0.5

        if -1.0 <= numeric_value <= 1.0:
            normalized = (numeric_value + 1.0) / 2.0
        else:
            normalized = numeric_value
        return float(min(1.0, max(0.0, normalized)))

    def _normalize_risk_level(self, value: Optional[str]) -> RiskLevel:
        if isinstance(value, RiskLevel):
            return value
        try:
            return RiskLevel(str(value))
        except Exception:
            return RiskLevel.MEDIUM

    def _normalize_time_horizon(self, value: Optional[str]) -> TimeHorizon:
        if isinstance(value, TimeHorizon):
            return value
        try:
            return TimeHorizon(str(value))
        except Exception:
            return TimeHorizon.MEDIUM_TERM

    def _normalize_recommendation_type(
        self, value: Optional[str], sentiment_score: Optional[float]
    ) -> RecommendationType:
        if isinstance(value, RecommendationType):
            return value
        if value:
            try:
                return RecommendationType(str(value))
            except Exception:
                pass

        score = sentiment_score if sentiment_score is not None else 0.5
        if score >= 0.75:
            return RecommendationType.STRONG_BUY
        if score >= 0.6:
            return RecommendationType.BUY
        if score <= 0.25:
            return RecommendationType.STRONG_SELL
        if score <= 0.4:
            return RecommendationType.SELL
        return RecommendationType.HOLD

    def _compute_fundamental_score(self, financials: Dict[str, Any]) -> float:
        if not financials:
            return 0.55

        score = 0.55
        pe_ratio = financials.get("pe_ratio")
        if pe_ratio and pe_ratio > 0:
            if pe_ratio < 15:
                score += 0.1
            elif pe_ratio < 30:
                score += 0.05
            else:
                score -= 0.05

        roe = financials.get("return_on_equity")
        if roe is not None:
            score += min(max(roe, 0.0), 0.3)

        debt_to_equity = financials.get("debt_to_equity")
        if debt_to_equity:
            score -= min(max(debt_to_equity - 1.0, 0.0) * 0.05, 0.15)

        revenue = financials.get("revenue")
        net_income = financials.get("net_income")
        if revenue and net_income:
            margin = net_income / revenue
            score += min(max(margin, 0.0), 0.2)

        return float(min(0.95, max(0.3, score)))

    def _compute_technical_score(self, technicals: Dict[str, Any]) -> float:
        if not technicals:
            return 0.55

        score = 0.5

        rsi = technicals.get("rsi")
        if rsi is not None:
            if 40 <= rsi <= 70:
                score += 0.1
            elif rsi < 30 or rsi > 80:
                score -= 0.1

        macd = technicals.get("macd")
        macd_signal = technicals.get("macd_signal")
        if macd is not None and macd_signal is not None:
            score += 0.05 if macd > macd_signal else -0.05

        sma_20 = technicals.get("sma_20")
        sma_50 = technicals.get("sma_50")
        if sma_20 is not None and sma_50 is not None:
            score += 0.05 if sma_20 >= sma_50 else -0.05

        return float(min(0.9, max(0.3, score)))

    def _clamp_score(self, value: Optional[float], fallback: float) -> float:
        base = value if value is not None else fallback
        try:
            base = float(base)
        except (TypeError, ValueError):
            base = fallback if fallback is not None else 0.5
        return float(min(0.95, max(0.0, base)))

    def _ensure_list(self, value: Optional[Any]) -> List[str]:
        if value is None:
            return []
        if isinstance(value, list):
            return [str(item) for item in value if item is not None]
        return [str(value)]

    def _extract_current_price(
        self,
        market_data: Dict[str, Any],
        fundamentals: Dict[str, Any],
        technicals: Dict[str, Any],
    ) -> float:
        for source in (market_data, fundamentals, technicals):
            price = source.get("current_price")
            if price is not None:
                try:
                    return float(price)
                except (TypeError, ValueError):
                    continue
        return 100.0

    def _estimate_target_price(
        self,
        current_price: float,
        recommendation_type: RecommendationType,
        sentiment_score: float,
    ) -> float:
        delta = self._estimate_target_delta(recommendation_type)
        adjustment = (sentiment_score - 0.5) * 0.1
        target = current_price * (1 + delta + adjustment)
        return max(target, 1.0)

    def _estimate_target_delta(self, recommendation_type: RecommendationType) -> float:
        deltas = {
            RecommendationType.STRONG_BUY: 0.2,
            RecommendationType.BUY: 0.12,
            RecommendationType.HOLD: 0.04,
            RecommendationType.SELL: -0.05,
            RecommendationType.STRONG_SELL: -0.1,
        }
        return deltas.get(recommendation_type, 0.05)

    def _estimate_stop_loss(
        self,
        current_price: float,
        risk_level: RiskLevel,
        recommendation_type: RecommendationType,
    ) -> Optional[float]:
        risk_percents = {
            RiskLevel.LOW: 0.04,
            RiskLevel.MEDIUM: 0.07,
            RiskLevel.HIGH: 0.1,
            RiskLevel.VERY_HIGH: 0.15,
        }
        pct = risk_percents.get(risk_level, 0.07)

        if recommendation_type in [
            RecommendationType.SELL,
            RecommendationType.STRONG_SELL,
        ]:
            return current_price * (1 + pct)
        return current_price * (1 - pct)

    def _map_expected_volatility(self, risk_level: RiskLevel) -> Optional[float]:
        mapping = {
            RiskLevel.LOW: 0.08,
            RiskLevel.MEDIUM: 0.12,
            RiskLevel.HIGH: 0.18,
            RiskLevel.VERY_HIGH: 0.25,
        }
        return mapping.get(risk_level)

    def _categorize_market_cap(self, value: Optional[float]) -> Optional[str]:
        if value is None:
            return None
        try:
            market_cap = float(value)
        except (TypeError, ValueError):
            return None

        if market_cap >= 200_000_000_000:
            return "mega_cap"
        if market_cap >= 10_000_000_000:
            return "large_cap"
        if market_cap >= 2_000_000_000:
            return "mid_cap"
        if market_cap >= 300_000_000:
            return "small_cap"
        return "micro_cap"

    def _get_article_symbols(self, article: Any) -> List[str]:
        if isinstance(article, dict):
            symbols = article.get("symbols", [])
        else:
            symbols = getattr(article, "symbols", [])

        if isinstance(symbols, list):
            return [str(symbol) for symbol in symbols if symbol is not None]
        if symbols is None:
            return []
        return [str(symbols)]

    def _get_article_value(
        self, article: Any, key: str, default: Optional[str] = None
    ) -> Any:
        if isinstance(article, dict):
            return article.get(key, default)
        return getattr(article, key, default)

    def _article_to_dict(self, article: Any) -> Dict[str, Any]:
        if isinstance(article, dict):
            return article
        if hasattr(article, "model_dump"):
            return article.model_dump()
        if hasattr(article, "dict"):
            return article.dict()
        return {
            "title": self._get_article_value(article, "title", ""),
            "url": self._get_article_value(article, "url", ""),
            "content": self._get_article_value(article, "content", ""),
            "source": self._get_article_value(article, "source", ""),
            "published_at": self._get_article_value(
                article, "published_at", datetime.now().isoformat()
            ),
            "symbols": self._get_article_symbols(article),
        }
