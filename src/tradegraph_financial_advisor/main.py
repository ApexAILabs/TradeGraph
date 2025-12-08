import argparse
import asyncio
import json
import sys
from datetime import datetime
from typing import Any, Dict, List, Optional

from loguru import logger

from .workflows.analysis_workflow import FinancialAnalysisWorkflow
from .agents.recommendation_engine import TradingRecommendationEngine
from .agents.report_analysis_agent import ReportAnalysisAgent
from .config.settings import settings
from .services.db_manager import db_manager
from .utils.helpers import save_analysis_results
from .utils.symbols import resolve_symbol
from .visualization import charts


class FinancialAdvisor:
    def __init__(self, llm_model_name: str = "gpt-5-nano"):
        self.llm_model_name = llm_model_name
        self.workflow = FinancialAnalysisWorkflow(llm_model_name=self.llm_model_name)
        self.recommendation_engine = TradingRecommendationEngine(
            model_name=self.llm_model_name
        )
        self.report_analyzer = ReportAnalysisAgent(llm_model_name=self.llm_model_name)

    async def analyze_portfolio(
        self,
        symbols: List[str],
        portfolio_size: float = None,
        risk_tolerance: str = "medium",
        time_horizon: str = "medium_term",
        include_reports: bool = True,
    ) -> Dict[str, Any]:
        """
        Perform comprehensive portfolio analysis and generate recommendations.

        Args:
            symbols: List of stock symbols to analyze
            portfolio_size: Portfolio size in dollars
            risk_tolerance: "conservative", "medium", or "aggressive"
            time_horizon: "short_term", "medium_term", or "long_term"
            include_reports: Whether to include SEC filing analysis

        Returns:
            Complete analysis results including recommendations
        """
        try:
            logger.info(f"Starting comprehensive analysis for {len(symbols)} symbols")

            if portfolio_size is None:
                portfolio_size = settings.default_portfolio_size

            symbol_mappings: Dict[str, Dict[str, str]] = {}
            asset_mix = {"equity": 0, "crypto": 0}
            base_symbols: List[str] = []
            for symbol in symbols:
                resolution = resolve_symbol(symbol)
                symbol_mappings[symbol] = resolution
                asset_mix[resolution["asset_type"]] += 1
                base_symbols.append(resolution["base_symbol"])

            # Step 1: Run the main workflow
            workflow_results = await self.workflow.analyze_portfolio(
                symbols=symbols,
                portfolio_size=portfolio_size,
                risk_tolerance=risk_tolerance,
                time_horizon=time_horizon,
            )

            portfolio_recommendation = workflow_results.get("portfolio_recommendation")
            sentiment_analysis = workflow_results.get("sentiment_analysis", {})
            news_data = workflow_results.get("news_data", {})
            financial_data = workflow_results.get("financial_data", {})
            recommendations = workflow_results.get("recommendations", [])
            analysis_context = workflow_results.get("analysis_context", {})

            # Step 2: Enhance with detailed report analysis if requested
            report_analyses = {}
            if include_reports:
                try:
                    await self.report_analyzer.start()

                    report_input = {
                        "symbols": symbols,
                        "report_types": ["10-K", "10-Q"],
                        "analysis_depth": "detailed",
                    }

                    report_result = await self.report_analyzer.execute(report_input)
                    report_analyses = report_result.get("report_analysis", {})

                    await self.report_analyzer.stop()

                except Exception as e:
                    logger.warning(f"Report analysis failed: {str(e)}")

            # Step 3: Gather database-backed insights
            stored_price_history: Dict[str, Any] = {}
            for symbol, resolution in symbol_mappings.items():
                history = db_manager.get_stock_history(resolution["resolved_symbol"])
                if history:
                    stored_price_history[symbol] = history

            knowledge_graph_articles = db_manager.get_recent_articles(base_symbols, limit=15)
            recent_queries = db_manager.get_recent_queries(limit=5)

            advisor_report = self._build_advisor_report(
                recommendations=recommendations,
                sentiment_analysis=sentiment_analysis,
                portfolio=portfolio_recommendation,
                news_data=news_data,
            )

            # Combine all results
            final_results = {
                "analysis_summary": {
                    "symbols_analyzed": symbols,
                    "portfolio_size": portfolio_size,
                    "risk_tolerance": risk_tolerance,
                    "time_horizon": time_horizon,
                    "analysis_timestamp": datetime.now().isoformat(),
                    "asset_breakdown": asset_mix,
                    "symbol_metadata": symbol_mappings,
                },
                "portfolio_recommendation": (
                    portfolio_recommendation if portfolio_recommendation else None
                ),
                "sentiment_analysis": sentiment_analysis,
                "detailed_reports": report_analyses,
                "news_data": news_data,
                "financial_data": financial_data,
                "recommendations": recommendations,
                "analysis_context": analysis_context,
                "database_insights": {
                    "price_history": stored_price_history,
                    "knowledge_graph_articles": knowledge_graph_articles,
                    "recent_queries": recent_queries,
                },
                "advisor_report": advisor_report,
                "analysis_metadata": {
                    "workflow_version": "1.0.0",
                    "agents_used": [
                        "NewsReaderAgent",
                        "FinancialAnalysisAgent",
                        "ReportAnalysisAgent",
                    ],
                    "data_sources": settings.news_sources,
                    "total_analysis_time": "calculated_at_runtime",
                },
            }

            try:
                summary_payload = json.dumps(
                    {
                        "symbols": symbols,
                        "risk": portfolio_recommendation.get("overall_risk_level")
                        if portfolio_recommendation
                        else None,
                        "recommendations": [
                            {
                                "symbol": rec.get("symbol"),
                                "action": rec.get("recommendation"),
                                "confidence": rec.get("confidence_score"),
                            }
                            for rec in recommendations
                        ],
                    }
                )
                db_manager.log_query(
                    query_text=", ".join(symbols),
                    response_summary=summary_payload[:4000],
                )
            except Exception as logging_error:
                logger.warning(f"Failed to log query in DuckDB: {logging_error}")

            logger.info("Comprehensive analysis completed successfully")
            return final_results

        except Exception as e:
            logger.error(f"Portfolio analysis failed: {str(e)}")
            raise

    async def quick_analysis(
        self, symbols: List[str], analysis_type: str = "standard"
    ) -> Dict[str, Any]:
        """
        Perform quick analysis without full report scraping.

        Args:
            symbols: List of stock symbols
            analysis_type: "basic", "standard", or "detailed"

        Returns:
            Quick analysis results
        """
        try:
            logger.info(f"Starting quick analysis for {symbols}")

            if analysis_type == "basic":
                # Basic analysis - just market data and news
                portfolio_rec = await self.workflow.analyze_portfolio(
                    symbols=symbols,
                    portfolio_size=50000,  # Default smaller size for quick analysis
                    risk_tolerance="medium",
                )

                return {
                    "analysis_type": "basic",
                    "symbols": symbols,
                    "recommendations": (
                        [rec.dict() for rec in portfolio_rec.recommendations]
                        if portfolio_rec
                        else []
                    ),
                    "analysis_timestamp": datetime.now().isoformat(),
                }

            elif analysis_type == "standard":
                return await self.analyze_portfolio(
                    symbols=symbols, include_reports=False
                )

            else:  # detailed
                return await self.analyze_portfolio(
                    symbols=symbols, include_reports=True
                )

        except Exception as e:
            logger.error(f"Quick analysis failed: {str(e)}")
            raise

    async def get_stock_alerts(self, symbols: List[str]) -> List[Dict[str, Any]]:
        """
        Generate real-time alerts for given symbols.

        Args:
            symbols: List of stock symbols to monitor

        Returns:
            List of current alerts
        """
        try:
            logger.info(f"Generating alerts for {symbols}")

            # This would typically connect to a real-time data feed
            # For demo purposes, using current analysis
            await self.quick_analysis(symbols, "basic")

            alerts = []
            # Extract alerts from analysis (simplified)
            for symbol in symbols:
                alerts.append(
                    {
                        "symbol": symbol,
                        "alert_type": "analysis_available",
                        "message": f"Analysis completed for {symbol}",
                        "timestamp": datetime.now().isoformat(),
                        "urgency": "low",
                    }
                )

            return alerts

        except Exception as e:
            logger.error(f"Alert generation failed: {str(e)}")
            return []

    def _build_advisor_report(
        self,
        recommendations: List[Dict[str, Any]],
        sentiment_analysis: Dict[str, Any],
        portfolio: Optional[Dict[str, Any]],
        news_data: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        if not recommendations:
            return None

        buy_set = {"buy", "strong_buy"}
        sell_set = {"sell", "strong_sell"}
        buy_count = sum(
            1 for rec in recommendations if rec.get("recommendation") in buy_set
        )
        sell_count = sum(
            1 for rec in recommendations if rec.get("recommendation") in sell_set
        )
        hold_count = len(recommendations) - buy_count - sell_count

        if buy_count > sell_count and buy_count >= hold_count:
            stance = "bullish"
        elif sell_count > buy_count:
            stance = "defensive"
        else:
            stance = "balanced"

        risk_mapping = {"low": 0.25, "medium": 0.5, "high": 0.75, "very_high": 0.9}
        avg_risk = 0.0
        if recommendations:
            avg_risk = sum(
                risk_mapping.get(rec.get("risk_level", "medium"), 0.5)
                for rec in recommendations
            ) / len(recommendations)

        if avg_risk < 0.35:
            risk_label = "low"
        elif avg_risk < 0.6:
            risk_label = "medium"
        else:
            risk_label = "high"

        positions = []
        for rec in recommendations:
            notes = rec.get("analyst_notes") or "; ".join(rec.get("key_factors", [])[:3])
            positions.append(
                {
                    "symbol": rec.get("symbol"),
                    "action": rec.get("recommendation", "").upper(),
                    "confidence": rec.get("confidence_score"),
                    "allocation": rec.get("recommended_allocation"),
                    "target_price": rec.get("target_price"),
                    "stop_loss": rec.get("stop_loss"),
                    "risk": rec.get("risk_level"),
                    "should_buy": rec.get("recommendation") in buy_set,
                    "notes": notes,
                    "catalysts": rec.get("catalysts", [])[:3],
                }
            )

        sentiment_sections = []
        for symbol, summary in sentiment_analysis.items():
            sentiment_sections.append(
                {
                    "symbol": symbol,
                    "label": summary.get("sentiment_label", "neutral"),
                    "score": summary.get("sentiment_score"),
                    "confidence": summary.get("confidence"),
                    "news_summary": summary.get("news_summary"),
                    "article_count": summary.get("article_count", 0),
                }
            )

        articles = (news_data or {}).get("articles", [])
        news_highlights = []
        for article in articles[:5]:
            news_highlights.append(
                {
                    "title": article.get("title"),
                    "source": article.get("source"),
                    "sentiment": article.get("sentiment"),
                    "impact_score": article.get("impact_score"),
                    "url": article.get("url"),
                }
            )

        summary = (
            f"{len(recommendations)} positions analyzed. "
            f"Bias: {stance.upper()} | Portfolio risk: {risk_label.upper()}"
        )
        if portfolio:
            expected_return = portfolio.get("expected_return")
            if isinstance(expected_return, (int, float)):
                summary += f" | Expected return {expected_return:.1%}"

        return {
            "stance": stance,
            "summary": summary,
            "positions": positions,
            "risk_summary": {
                "portfolio": portfolio.get("overall_risk_level") if portfolio else None,
                "position_risk": risk_label,
                "buy_signals": buy_count,
                "sell_signals": sell_count,
            },
            "sentiment_overview": sentiment_sections,
            "news_highlights": news_highlights,
        }

    def print_recommendations(self, results: Dict[str, Any]) -> None:
        """
        Pretty print analysis results to console.
        """
        print("\n" + "=" * 80)
        print("TRADEGRAPH FINANCIAL ADVISOR - ANALYSIS RESULTS")
        print("=" * 80)

        # Analysis Summary
        summary = results.get("analysis_summary", {})
        print(f"\nAnalysis Date: {summary.get('analysis_timestamp', 'Unknown')}")
        print(f"Symbols Analyzed: {', '.join(summary.get('symbols_analyzed', []))}")
        print(f"Portfolio Size: ${summary.get('portfolio_size', 0):,.2f}")
        print(f"Risk Tolerance: {summary.get('risk_tolerance', 'Unknown')}")

        # Portfolio Recommendation
        portfolio = results.get("portfolio_recommendation")
        if portfolio:
            print("\n📊 PORTFOLIO RECOMMENDATION")
            print(f"Overall Confidence: {portfolio.get('total_confidence', 0):.1%}")
            print(
                f"Diversification Score: {portfolio.get('diversification_score', 0):.1%}"
            )
            print(f"Risk Level: {portfolio.get('overall_risk_level', 'Unknown')}")

            recommendations = portfolio.get("recommendations", [])
            if recommendations:
                print(
                    f"\n📈 INDIVIDUAL RECOMMENDATIONS ({len(recommendations)} stocks):"
                )
                print("-" * 60)

                for rec in recommendations:
                    symbol = rec.get("symbol", "Unknown")
                    recommendation = rec.get("recommendation", "hold").upper()
                    confidence = rec.get("confidence_score", 0)
                    allocation = rec.get("recommended_allocation", 0)
                    target = rec.get("target_price")
                    current = rec.get("current_price", 0)

                    print(
                        f"\n{symbol}: {recommendation} (Confidence: {confidence:.1%})"
                    )
                    print(
                        f"  Current: ${current:.2f} | Target: ${target:.2f}"
                        if target
                        else f"  Current: ${current:.2f}"
                    )
                    print(
                        f"  Allocation: {allocation:.1%} | Risk: {rec.get('risk_level', 'Unknown')}"
                    )

                    factors = rec.get("key_factors", [])
                    if factors:
                        print(f"  Key Factors: {', '.join(factors[:2])}")

        # Detailed Reports Summary
        reports = results.get("detailed_reports", {})
        if reports:
            print("\n📋 DETAILED REPORT ANALYSIS")
            print("-" * 40)

            for symbol, report in reports.items():
                if "error" not in report:
                    health_score = report.get("financial_health_score", 0)
                    print(f"\n{symbol} - Financial Health: {health_score:.1f}/10")

                    summary_text = report.get("executive_summary", "")
                    if summary_text:
                        print(f"  Summary: {summary_text[:100]}...")

        print("\n" + "=" * 80)


async def main():
    """
    Command-line interface for TradeGraph Financial Advisor.
    """
    parser = argparse.ArgumentParser(
        description="TradeGraph Financial Advisor - AI-powered investment analysis"
    )
    parser.add_argument(
        "symbols", nargs="+", help="Stock symbols to analyze (e.g., AAPL MSFT GOOGL)"
    )
    parser.add_argument(
        "--portfolio-size",
        type=float,
        default=None,
        help="Portfolio size in dollars (default: from config)",
    )
    parser.add_argument(
        "--risk-tolerance",
        choices=["conservative", "medium", "aggressive"],
        default="medium",
        help="Risk tolerance level",
    )
    parser.add_argument(
        "--time-horizon",
        choices=["short_term", "medium_term", "long_term"],
        default="medium_term",
        help="Investment time horizon",
    )
    parser.add_argument(
        "--analysis-type",
        choices=["quick", "standard", "comprehensive"],
        default="standard",
        help="Analysis depth",
    )
    parser.add_argument(
        "--output-format",
        choices=["console", "json"],
        default="console",
        help="Output format",
    )
    parser.add_argument(
        "--alerts-only", action="store_true", help="Generate alerts only"
    )

    args = parser.parse_args()

    # Configure logging
    logger.remove()  # Remove default handler
    logger.add(
        sys.stderr,
        level=settings.log_level,
        format="<green>{time}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
    )

    try:
        advisor = FinancialAdvisor()

        if args.alerts_only:
            # Generate alerts only
            alerts = await advisor.get_stock_alerts(args.symbols)

            # Save alerts to JSON file with timestamp
            try:
                # Create results structure for alerts
                alerts_results = {
                    "analysis_summary": {
                        "symbols_analyzed": args.symbols,
                        "analysis_type": "alerts_only",
                        "analysis_timestamp": datetime.now().isoformat(),
                    },
                    "alerts": alerts,
                }
                filepath = save_analysis_results(alerts_results)
                logger.info(f"Alerts results automatically saved to: {filepath}")
            except Exception as e:
                logger.warning(f"Failed to save alerts file: {str(e)}")

            if args.output_format == "json":
                import json

                print(json.dumps(alerts, indent=2))
            else:
                print("\n🚨 CURRENT ALERTS:")
                for alert in alerts:
                    print(f"  {alert['symbol']}: {alert['message']}")

        else:
            # Full analysis
            if args.analysis_type == "quick":
                results = await advisor.quick_analysis(args.symbols, "basic")
            elif args.analysis_type == "comprehensive":
                results = await advisor.analyze_portfolio(
                    symbols=args.symbols,
                    portfolio_size=args.portfolio_size,
                    risk_tolerance=args.risk_tolerance,
                    time_horizon=args.time_horizon,
                    include_reports=True,
                )
            else:  # standard
                results = await advisor.analyze_portfolio(
                    symbols=args.symbols,
                    portfolio_size=args.portfolio_size,
                    risk_tolerance=args.risk_tolerance,
                    time_horizon=args.time_horizon,
                    include_reports=False,
                )

            # Always save results to JSON file with timestamp
            try:
                filepath = save_analysis_results(results)
                logger.info(f"Analysis results automatically saved to: {filepath}")
            except Exception as e:
                logger.warning(f"Failed to save results file: {str(e)}")

            # Generate portfolio allocation chart
            try:
                portfolio_rec = results.get("portfolio_recommendation")

                if portfolio_rec and portfolio_rec.get("recommendations"):
                    recommendations = portfolio_rec.get("recommendations", [])

                    chart_path = charts.create_portfolio_allocation_chart(
                        recommendations=recommendations,
                        output_path="results/portfolio_allocation.html",
                    )

                    logger.info(f"Portfolio allocation chart saved to: {chart_path}")
                else:
                    logger.warning(
                        "No recommendations found to create allocation chart"
                    )

            except Exception as e:
                logger.warning(
                    f"Failed to generate portfolio allocation chart: {str(e)}"
                )

            # Display results based on output format
            if args.output_format == "json":
                import json

                print(json.dumps(results, indent=2, default=str))
            else:
                advisor.print_recommendations(results)

    except KeyboardInterrupt:
        logger.info("Analysis interrupted by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Analysis failed: {str(e)}")
        sys.exit(1)


def cli_main():
    """
    Entry point for the tradegraph command-line script.
    """
    asyncio.run(main())


if __name__ == "__main__":
    cli_main()
