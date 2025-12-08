import json
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime
import numpy as np
from loguru import logger
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

from .base_agent import BaseAgent
from ..models.recommendations import (
    TradingRecommendation,
    PortfolioRecommendation,
    RecommendationType,
    RiskLevel,
    TimeHorizon,
    AlertRecommendation,
)
from ..config.settings import settings


class TradingRecommendationEngine(BaseAgent):
    def __init__(self, model_name: str = "gpt-5-nano", **kwargs):
        super().__init__(
            name="TradingRecommendationEngine",
            description="Generates sophisticated trading recommendations using multi-factor analysis",
            **kwargs,
        )
        self.model_name = model_name
        self.llm = ChatOpenAI(
            model=self.model_name, temperature=0.1, api_key=settings.openai_api_key
        )

    async def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        analysis_contexts = input_data.get("analysis_contexts", {})
        portfolio_constraints = input_data.get("portfolio_constraints", {})
        risk_preferences = input_data.get("risk_preferences", {})

        logger.info("Generating trading recommendations")

        recommendations = []
        alerts = []

        for symbol, context in analysis_contexts.items():
            try:
                # Generate individual recommendation
                recommendation = await self._generate_stock_recommendation(
                    symbol, context, risk_preferences
                )
                if recommendation:
                    recommendations.append(recommendation)

                # Generate alerts if needed
                symbol_alerts = await self._generate_alerts(symbol, context)
                alerts.extend(symbol_alerts)

            except Exception as e:
                logger.error(f"Error generating recommendation for {symbol}: {str(e)}")

        # Create portfolio optimization
        portfolio_recommendation = await self._optimize_portfolio(
            recommendations, portfolio_constraints, risk_preferences
        )

        return {
            "individual_recommendations": [rec.dict() for rec in recommendations],
            "portfolio_recommendation": (
                portfolio_recommendation.dict() if portfolio_recommendation else None
            ),
            "alerts": [alert.dict() for alert in alerts],
            "generation_timestamp": datetime.now().isoformat(),
        }

    async def _generate_stock_recommendation(
        self, symbol: str, context: Dict[str, Any], risk_preferences: Dict[str, Any]
    ) -> Optional[TradingRecommendation]:
        try:
            # Extract data from context
            market_data = context.get("market_data", {})
            financials = context.get("financials", {})
            technical_data = context.get("technical_indicators", {})
            news_sentiment = context.get("sentiment_analysis", {})
            report_analysis = context.get("report_analysis", {})

            # Calculate individual scores
            fundamental_score = await self._calculate_fundamental_score(financials, report_analysis)
            technical_score = await self._calculate_technical_score(technical_data, market_data)
            sentiment_score = await self._calculate_sentiment_score(news_sentiment)

            # Overall confidence score (weighted combination)
            weights = {"fundamental": 0.4, "technical": 0.3, "sentiment": 0.3}
            confidence_score = (
                weights["fundamental"] * fundamental_score
                + weights["technical"] * technical_score
                + weights["sentiment"] * sentiment_score
            )

            # Determine recommendation type
            recommendation_type = self._determine_recommendation_type(
                confidence_score, fundamental_score, technical_score, sentiment_score
            )

            # Calculate risk level
            risk_level = self._calculate_risk_level(
                market_data, financials, technical_data, report_analysis
            )

            # Determine time horizon
            time_horizon = self._determine_time_horizon(
                recommendation_type, risk_level, technical_data
            )

            # Calculate position sizing
            recommended_allocation = self._calculate_position_size(
                confidence_score, risk_level, risk_preferences, recommendation_type
            )

            # Calculate target price and stop loss
            current_price = market_data.get("current_price", 100.0)
            target_price, stop_loss = await self._calculate_price_targets(
                symbol, current_price, recommendation_type, technical_data, financials
            )

            # Generate supporting factors
            factors = await self._generate_key_factors(
                symbol, financials, technical_data, news_sentiment, report_analysis
            )

            recommendation = TradingRecommendation(
                symbol=symbol,
                company_name=financials.get("company_name", symbol),
                recommendation=recommendation_type,
                confidence_score=confidence_score,
                target_price=target_price,
                stop_loss=stop_loss,
                current_price=current_price,
                risk_level=risk_level,
                time_horizon=time_horizon,
                recommended_allocation=recommended_allocation,
                fundamental_score=fundamental_score,
                technical_score=technical_score,
                sentiment_score=sentiment_score,
                key_factors=factors["key_factors"],
                risks=factors["risks"],
                catalysts=factors["catalysts"],
                analyst_notes=factors["analyst_notes"],
                sector=financials.get("sector", "Unknown"),
                expected_return=await self._calculate_expected_return(
                    current_price, target_price, recommendation_type
                ),
            )

            return recommendation

        except Exception as e:
            logger.error(f"Error generating recommendation for {symbol}: {str(e)}")
            return None

    async def _calculate_fundamental_score(
        self, financials: Dict[str, Any], report_analysis: Dict[str, Any]
    ) -> float:
        try:
            score = 0.5  # Base score

            # P/E ratio analysis
            pe_ratio = financials.get("pe_ratio")
            if pe_ratio:
                if 0 < pe_ratio < 15:
                    score += 0.1  # Undervalued
                elif 15 <= pe_ratio < 25:
                    score += 0.05  # Fair value
                elif pe_ratio >= 35:
                    score -= 0.1  # Overvalued

            # ROE analysis
            roe = financials.get("return_on_equity")
            if roe:
                if roe > 0.15:
                    score += 0.1
                elif roe > 0.10:
                    score += 0.05
                elif roe < 0:
                    score -= 0.15

            # Debt analysis
            debt_ratio = financials.get("debt_to_equity")
            if debt_ratio:
                if debt_ratio < 0.3:
                    score += 0.05
                elif debt_ratio > 1.0:
                    score -= 0.1

            # Growth metrics
            revenue_growth = financials.get("revenue_growth")
            if revenue_growth and revenue_growth > 0.1:
                score += 0.1

            # Report analysis integration
            if report_analysis:
                health_score = report_analysis.get("financial_health_score", 5.0)
                score += (health_score - 5.0) / 10.0  # Normalize to -0.5 to +0.5

            return max(0.0, min(1.0, score))

        except Exception as e:
            logger.warning(f"Error calculating fundamental score: {str(e)}")
            return 0.5

    async def _calculate_technical_score(
        self, technical_data: Dict[str, Any], market_data: Dict[str, Any]
    ) -> float:
        try:
            score = 0.5  # Base score

            current_price = market_data.get("current_price", 100.0)

            # Moving average analysis
            sma_20 = technical_data.get("sma_20")
            sma_50 = technical_data.get("sma_50")

            if sma_20 and sma_50:
                if current_price > sma_20 > sma_50:
                    score += 0.15  # Strong uptrend
                elif current_price > sma_20:
                    score += 0.1  # Uptrend
                elif current_price < sma_20 < sma_50:
                    score -= 0.15  # Strong downtrend
                elif current_price < sma_20:
                    score -= 0.1  # Downtrend

            # RSI analysis
            rsi = technical_data.get("rsi")
            if rsi:
                if rsi < 30:
                    score += 0.1  # Oversold
                elif rsi > 70:
                    score -= 0.1  # Overbought
                elif 40 <= rsi <= 60:
                    score += 0.05  # Neutral momentum

            # MACD analysis
            macd = technical_data.get("macd")
            macd_signal = technical_data.get("macd_signal")
            if macd and macd_signal:
                if macd > macd_signal:
                    score += 0.05  # Bullish signal
                else:
                    score -= 0.05  # Bearish signal

            # Volume analysis
            volume = market_data.get("volume", 0)
            if volume > 0:
                # This would need historical volume data for proper analysis
                # For now, just checking if volume exists
                score += 0.02

            # Support/Resistance analysis
            support = technical_data.get("support_level")
            resistance = technical_data.get("resistance_level")
            if support and resistance and current_price:
                if current_price <= support * 1.05:  # Near support
                    score += 0.05
                elif current_price >= resistance * 0.95:  # Near resistance
                    score -= 0.05

            return max(0.0, min(1.0, score))

        except Exception as e:
            logger.warning(f"Error calculating technical score: {str(e)}")
            return 0.5

    async def _calculate_sentiment_score(self, sentiment_data: Dict[str, Any]) -> float:
        try:
            if not sentiment_data:
                return 0.5

            sentiment_score = sentiment_data.get("sentiment_score", 0.0)
            confidence = sentiment_data.get("confidence", 0.5)

            # Convert sentiment score (-1 to 1) to 0 to 1 scale
            normalized_score = (sentiment_score + 1) / 2

            # Weight by confidence
            weighted_score = 0.5 + (normalized_score - 0.5) * confidence

            return max(0.0, min(1.0, weighted_score))

        except Exception as e:
            logger.warning(f"Error calculating sentiment score: {str(e)}")
            return 0.5

    def _determine_recommendation_type(
        self,
        confidence_score: float,
        fundamental_score: float,
        technical_score: float,
        sentiment_score: float,
    ) -> RecommendationType:
        if confidence_score >= 0.8:
            return RecommendationType.STRONG_BUY
        elif confidence_score >= 0.65:
            return RecommendationType.BUY
        elif confidence_score >= 0.55:
            return RecommendationType.HOLD
        elif confidence_score >= 0.35:
            return RecommendationType.HOLD
        elif confidence_score >= 0.2:
            return RecommendationType.SELL
        else:
            return RecommendationType.STRONG_SELL

    def _calculate_risk_level(
        self,
        market_data: Dict[str, Any],
        financials: Dict[str, Any],
        technical_data: Dict[str, Any],
        report_analysis: Dict[str, Any],
    ) -> RiskLevel:
        risk_score = 0.0

        # Beta risk
        beta = financials.get("beta", 1.0)
        if beta > 1.5:
            risk_score += 1
        elif beta > 1.2:
            risk_score += 0.5

        # Debt risk
        debt_ratio = financials.get("debt_to_equity", 0.5)
        if debt_ratio > 1.0:
            risk_score += 1
        elif debt_ratio > 0.6:
            risk_score += 0.5

        # Volatility risk
        rsi = technical_data.get("rsi", 50)
        if rsi < 25 or rsi > 75:
            risk_score += 0.5

        # Market cap risk
        market_cap = financials.get("market_cap", 1e9)
        if market_cap < 1e9:  # Small cap
            risk_score += 1
        elif market_cap < 10e9:  # Mid cap
            risk_score += 0.5

        # Report analysis risk factors
        if report_analysis:
            risk_factors = report_analysis.get("risk_factors", [])
            risk_score += len(risk_factors) * 0.2

        if risk_score >= 2.5:
            return RiskLevel.VERY_HIGH
        elif risk_score >= 1.5:
            return RiskLevel.HIGH
        elif risk_score >= 0.75:
            return RiskLevel.MEDIUM
        else:
            return RiskLevel.LOW

    def _determine_time_horizon(
        self,
        recommendation_type: RecommendationType,
        risk_level: RiskLevel,
        technical_data: Dict[str, Any],
    ) -> TimeHorizon:
        # Technical momentum suggests shorter term
        rsi = technical_data.get("rsi", 50)
        if rsi < 30 or rsi > 70:
            return TimeHorizon.SHORT_TERM

        # High risk suggests shorter holding period
        if risk_level == RiskLevel.VERY_HIGH:
            return TimeHorizon.SHORT_TERM

        # Strong recommendations can be longer term
        if recommendation_type in [RecommendationType.STRONG_BUY, RecommendationType.STRONG_SELL]:
            return TimeHorizon.LONG_TERM

        return TimeHorizon.MEDIUM_TERM

    def _calculate_position_size(
        self,
        confidence_score: float,
        risk_level: RiskLevel,
        risk_preferences: Dict[str, Any],
        recommendation_type: RecommendationType,
    ) -> float:

        RECOMMENDATION_WEIGHTS = {
            RecommendationType.STRONG_BUY: 2.0,
            RecommendationType.BUY: 1.5,
            RecommendationType.HOLD: 0.8,
            RecommendationType.SELL: 0.3,
            RecommendationType.STRONG_SELL: 0.1,
        }

        base_allocation = confidence_score * RECOMMENDATION_WEIGHTS[recommendation_type]

        # Adjust for risk level
        risk_multipliers = {
            RiskLevel.LOW: 1.5,
            RiskLevel.MEDIUM: 1.0,
            RiskLevel.HIGH: 0.7,
            RiskLevel.VERY_HIGH: 0.4,
        }

        risk_adjusted = base_allocation * risk_multipliers[risk_level]

        # Adjust for user risk tolerance
        risk_tolerance = risk_preferences.get("risk_tolerance", "medium")
        tolerance_multipliers = {"conservative": 0.6, "medium": 1.0, "aggressive": 1.4}

        final_allocation = risk_adjusted * tolerance_multipliers.get(risk_tolerance, 1.0)

        return max(
            0.01, final_allocation
        )  # was Between 1% and 25% - changed for the new optimization logic

    async def _calculate_price_targets(
        self,
        symbol: str,
        current_price: float,
        recommendation_type: RecommendationType,
        technical_data: Dict[str, Any],
        financials: Dict[str, Any],
    ) -> Tuple[Optional[float], Optional[float]]:
        try:
            target_price = None
            stop_loss = None

            if recommendation_type in [RecommendationType.BUY, RecommendationType.STRONG_BUY]:
                # Calculate upside target
                resistance = technical_data.get("resistance_level")
                if resistance:
                    target_price = resistance * 1.05  # 5% above resistance
                else:
                    # Use P/E based target
                    pe_ratio = financials.get("pe_ratio")
                    if pe_ratio and pe_ratio > 0:
                        sector_avg_pe = 18  # Rough market average
                        if pe_ratio < sector_avg_pe:
                            target_price = current_price * (sector_avg_pe / pe_ratio)

                if not target_price:
                    target_price = current_price * 1.20  # 20% upside default

                # Calculate stop loss
                support = technical_data.get("support_level")
                if support:
                    stop_loss = support * 0.95  # 5% below support
                else:
                    stop_loss = current_price * 0.90  # 10% stop loss

            elif recommendation_type in [RecommendationType.SELL, RecommendationType.STRONG_SELL]:
                # For short positions
                support = technical_data.get("support_level")
                if support:
                    target_price = support * 0.95
                else:
                    target_price = current_price * 0.80  # 20% downside

                # Stop loss for short
                resistance = technical_data.get("resistance_level")
                if resistance:
                    stop_loss = resistance * 1.05
                else:
                    stop_loss = current_price * 1.10

            return target_price, stop_loss

        except Exception as e:
            logger.warning(f"Error calculating price targets for {symbol}: {str(e)}")
            return None, None

    async def _generate_key_factors(
        self,
        symbol: str,
        financials: Dict[str, Any],
        technical_data: Dict[str, Any],
        sentiment_data: Dict[str, Any],
        report_analysis: Dict[str, Any],
    ) -> Dict[str, List[str]]:
        try:
            analysis_prompt = f"""
            Based on the following analysis data for {symbol}, generate key investment factors:

            Financial Data: {str(financials)[:1000]}
            Technical Data: {str(technical_data)[:500]}
            Sentiment Data: {str(sentiment_data)[:500]}
            Report Analysis: {str(report_analysis)[:1000]}

            Generate factors in JSON format:
            {{
                "key_factors": ["5 most important positive factors"],
                "risks": ["5 most significant risks"],
                "catalysts": ["3-5 potential catalysts for price movement"],
                "analyst_notes": "Brief summary of investment thesis"
            }}
            """

            response = await self.llm.ainvoke([HumanMessage(content=analysis_prompt)])

            try:
                factors_data = json.loads(response.content)
                return factors_data
            except json.JSONDecodeError:
                # Fallback
                return {
                    "key_factors": ["Analysis available"],
                    "risks": ["Market volatility"],
                    "catalysts": ["Market sentiment"],
                    "analyst_notes": "Comprehensive analysis completed",
                }

        except Exception as e:
            logger.warning(f"Error generating factors for {symbol}: {str(e)}")
            return {
                "key_factors": ["Analysis available"],
                "risks": ["Market volatility"],
                "catalysts": ["Market sentiment"],
                "analyst_notes": "Analysis completed with limitations",
            }

    async def _calculate_expected_return(
        self,
        current_price: float,
        target_price: Optional[float],
        recommendation_type: RecommendationType,
    ) -> Optional[float]:
        if not target_price:
            return None

        if recommendation_type in [RecommendationType.BUY, RecommendationType.STRONG_BUY]:
            return (target_price - current_price) / current_price
        elif recommendation_type in [RecommendationType.SELL, RecommendationType.STRONG_SELL]:
            return (current_price - target_price) / current_price  # Profit from short

        return 0.0

    async def _optimize_portfolio(
        self,
        recommendations: List[TradingRecommendation],
        portfolio_constraints: Dict[str, Any],
        risk_preferences: Dict[str, Any],
    ) -> Optional[PortfolioRecommendation]:
        try:
            if not recommendations:
                return None

            portfolio_size = portfolio_constraints.get(
                "portfolio_size", settings.default_portfolio_size
            )
            max_positions = portfolio_constraints.get("max_positions", 10)

            # Select top recommendations
            sorted_recs = sorted(recommendations, key=lambda x: x.confidence_score, reverse=True)[
                :max_positions
            ]

            # Optimize allocations
            optimized_recs = await self._optimize_allocations(sorted_recs, portfolio_constraints)

            # Calculate portfolio metrics
            total_confidence = np.mean([rec.confidence_score for rec in optimized_recs])
            diversification_score = self._calculate_diversification_score(optimized_recs)

            # Risk assessment
            risk_levels = [rec.risk_level for rec in optimized_recs]
            overall_risk = self._determine_portfolio_risk(risk_levels)

            portfolio = PortfolioRecommendation(
                recommendations=optimized_recs,
                total_confidence=total_confidence,
                diversification_score=diversification_score,
                overall_risk_level=overall_risk,
                portfolio_size=portfolio_size,
                rebalancing_frequency="quarterly",
            )

            return portfolio

        except Exception as e:
            logger.error(f"Error optimizing portfolio: {str(e)}")
            return None

    async def _optimize_allocations(
        self, recommendations: List[TradingRecommendation], constraints: Dict[str, Any]
    ) -> List[TradingRecommendation]:
        total_allocation = sum(rec.recommended_allocation for rec in recommendations)

        if total_allocation == 0:
            return recommendations

        # scale all allocations proportionally to sum to 95%
        target_allocation = 0.95
        scale_factor = target_allocation / total_allocation

        for rec in recommendations:
            rec.recommended_allocation *= scale_factor

        # apply individual position cap (25% max per stock)
        # If any position exceeds 25%, redistribute the excess to others
        max_position = 0.25

        # Keep iterating until no position exceeds the cap
        max_iterations = 10
        for _ in range(max_iterations):
            excess = 0
            capped_count = 0

            # Calculate excess from capped positions
            for rec in recommendations:
                if rec.recommended_allocation > max_position:
                    excess += rec.recommended_allocation - max_position
                    rec.recommended_allocation = max_position
                    capped_count += 1

            # If no excess, its done
            if excess <= 0.0001:  # Small threshold for floating point
                break

            # Redistribute excess to uncapped positions proportionally
            uncapped_recs = [
                rec for rec in recommendations if rec.recommended_allocation < max_position
            ]

            if uncapped_recs:
                uncapped_total = sum(rec.recommended_allocation for rec in uncapped_recs)
                if uncapped_total > 0:
                    for rec in uncapped_recs:
                        # Distribute excess proportionally
                        rec.recommended_allocation += excess * (
                            rec.recommended_allocation / uncapped_total
                        )

        # Final normalization: if all positions got capped, scale up to reach target
        final_total = sum(rec.recommended_allocation for rec in recommendations)

        if final_total < target_allocation - 0.001:  # If its below target
            # Scale everything up proportionally to reach target
            final_scale = target_allocation / final_total
            for rec in recommendations:
                rec.recommended_allocation *= final_scale

        return recommendations

    def _calculate_diversification_score(
        self, recommendations: List[TradingRecommendation]
    ) -> float:
        if len(recommendations) <= 1:
            return 0.0

        # Simple diversification based on number of positions
        return min(1.0, len(recommendations) / 10.0)

    def _determine_portfolio_risk(self, risk_levels: List[RiskLevel]) -> RiskLevel:
        risk_values = {
            RiskLevel.LOW: 1,
            RiskLevel.MEDIUM: 2,
            RiskLevel.HIGH: 3,
            RiskLevel.VERY_HIGH: 4,
        }

        avg_risk = np.mean([risk_values[risk] for risk in risk_levels])

        if avg_risk >= 3.5:
            return RiskLevel.VERY_HIGH
        elif avg_risk >= 2.5:
            return RiskLevel.HIGH
        elif avg_risk >= 1.5:
            return RiskLevel.MEDIUM
        else:
            return RiskLevel.LOW

    async def _generate_alerts(
        self, symbol: str, context: Dict[str, Any]
    ) -> List[AlertRecommendation]:
        alerts = []

        try:
            market_data = context.get("market_data", {})
            technical_data = context.get("technical_indicators", {})

            current_price = market_data.get("current_price", 0)
            rsi = technical_data.get("rsi", 50)

            # RSI alerts
            if rsi < 25:
                alerts.append(
                    AlertRecommendation(
                        symbol=symbol,
                        alert_type="technical_oversold",
                        message=f"{symbol} RSI at {rsi:.1f} - potentially oversold",
                        urgency="medium",
                        trigger_conditions={"rsi": rsi, "threshold": 25},
                    )
                )
            elif rsi > 75:
                alerts.append(
                    AlertRecommendation(
                        symbol=symbol,
                        alert_type="technical_overbought",
                        message=f"{symbol} RSI at {rsi:.1f} - potentially overbought",
                        urgency="medium",
                        trigger_conditions={"rsi": rsi, "threshold": 75},
                    )
                )

            # Support/resistance alerts
            support = technical_data.get("support_level")
            resistance = technical_data.get("resistance_level")

            if support and current_price <= support * 1.02:
                alerts.append(
                    AlertRecommendation(
                        symbol=symbol,
                        alert_type="price_support",
                        message=f"{symbol} near support level at ${support:.2f}",
                        urgency="high",
                        trigger_conditions={"current_price": current_price, "support": support},
                    )
                )

            if resistance and current_price >= resistance * 0.98:
                alerts.append(
                    AlertRecommendation(
                        symbol=symbol,
                        alert_type="price_resistance",
                        message=f"{symbol} near resistance level at ${resistance:.2f}",
                        urgency="high",
                        trigger_conditions={
                            "current_price": current_price,
                            "resistance": resistance,
                        },
                    )
                )

        except Exception as e:
            logger.warning(f"Error generating alerts for {symbol}: {str(e)}")

        return alerts

    async def _health_check_impl(self) -> None:
        try:
            test_response = await self.llm.ainvoke(
                [HumanMessage(content="Respond with 'OK' for health check.")]
            )
            if "OK" not in test_response.content:
                raise Exception("LLM health check failed")
        except Exception as e:
            raise Exception(f"Recommendation engine health check failed: {str(e)}")
