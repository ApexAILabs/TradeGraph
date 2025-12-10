import asyncio
import sys
from typing import List, Dict, Any, Optional
from datetime import datetime
import argparse
from loguru import logger

from .workflows.analysis_workflow import FinancialAnalysisWorkflow
from .agents.recommendation_engine import TradingRecommendationEngine
from .agents.report_analysis_agent import ReportAnalysisAgent
from .agents.channel_report_agent import ChannelReportAgent
from .agents.multi_asset_allocation_agent import MultiAssetAllocationAgent
from .config.settings import settings, refresh_openai_api_key
from .utils.helpers import save_analysis_results
from .visualization import charts
from .services.channel_stream_service import FinancialNewsChannelService
from .services.price_trend_service import PriceTrendService
from .reporting import ChannelPDFReportWriter, MultiAssetPDFReportWriter


class FinancialAdvisor:
    def __init__(self, llm_model_name: str = "gpt-5-nano"):
        self.llm_model_name = llm_model_name
        self.workflow = FinancialAnalysisWorkflow(llm_model_name=self.llm_model_name)
        self.recommendation_engine = TradingRecommendationEngine(
            model_name=self.llm_model_name
        )
        self.report_analyzer = ReportAnalysisAgent(llm_model_name=self.llm_model_name)
        self.channel_report_agent = ChannelReportAgent(
            llm_model_name=self.llm_model_name
        )
        self.channel_service = FinancialNewsChannelService()
        self.trend_service = PriceTrendService()
        self.pdf_report_writer = ChannelPDFReportWriter()
        self.multi_asset_agent = MultiAssetAllocationAgent()
        self.multi_asset_pdf_writer = MultiAssetPDFReportWriter()

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

            # Step 1: Run the main workflow
            workflow_results = await self.workflow.analyze_portfolio(
                symbols=symbols,
                portfolio_size=portfolio_size,
                risk_tolerance=risk_tolerance,
                time_horizon=time_horizon,
            )

            portfolio_recommendation = workflow_results.get("portfolio_recommendation")
            sentiment_analysis = workflow_results.get("sentiment_analysis", {})

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

            # Step 3: Generate final recommendations
            analysis_contexts = {}
            for symbol in symbols:
                analysis_contexts[symbol] = {
                    "symbol": symbol,
                    "report_analysis": report_analyses.get(symbol, {}),
                    "portfolio_context": {
                        "portfolio_size": portfolio_size,
                        "risk_tolerance": risk_tolerance,
                        "time_horizon": time_horizon,
                    },
                }

            # Combine all results
            final_results = {
                "analysis_summary": {
                    "symbols_analyzed": symbols,
                    "portfolio_size": portfolio_size,
                    "risk_tolerance": risk_tolerance,
                    "time_horizon": time_horizon,
                    "analysis_timestamp": datetime.now().isoformat(),
                },
                "portfolio_recommendation": (
                    portfolio_recommendation if portfolio_recommendation else None
                ),
                "recommendations": workflow_results.get("recommendations", []),
                "sentiment_analysis": sentiment_analysis,
                "detailed_reports": report_analyses,
                "channel_streams": workflow_results.get("channel_streams", {}),
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
                        [rec.model_dump() for rec in portfolio_rec.recommendations]
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

    async def plan_multi_asset_allocation(
        self, *, budget: float, strategies: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        if budget <= 0:
            raise ValueError("Budget must be positive for allocation planning.")
        payload = {
            "budget": budget,
            "strategies": strategies,
        }
        return await self.multi_asset_agent.execute(payload)

    def build_multi_asset_pdf(
        self, plan: Dict[str, Any], output_path: Optional[str] = None
    ) -> str:
        return self.multi_asset_pdf_writer.build_report(
            plan=plan, output_path=output_path
        )

    async def generate_channel_pdf_report(
        self,
        symbols: List[str],
        *,
        portfolio_size: Optional[float] = None,
        risk_tolerance: str = "medium",
        time_horizon: str = "medium_term",
        include_reports: bool = False,
        existing_results: Optional[Dict[str, Any]] = None,
        output_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Create a PDF report that merges channel streams, recommendations, and trends."""

        reference_results = existing_results
        if reference_results is None:
            reference_results = await self.analyze_portfolio(
                symbols=symbols,
                portfolio_size=portfolio_size,
                risk_tolerance=risk_tolerance,
                time_horizon=time_horizon,
                include_reports=include_reports,
            )

        channel_streams = reference_results.get("channel_streams") or {}
        if not channel_streams:
            channel_streams = await self.channel_service.collect_all_channels(symbols)
            await self.channel_service.close()

        price_trends = await self.trend_service.get_trends_for_symbols(symbols)

        summary_payload = await self.channel_report_agent.execute(
            {
                "channel_payloads": channel_streams,
                "price_trends": price_trends,
                "recommendations": reference_results.get("recommendations", []),
            }
        )

        recommendations = reference_results.get("recommendations", [])
        portfolio_rec = reference_results.get("portfolio_recommendation")
        allocation_chart_path = None
        if recommendations:
            allocation_chart_path = charts.create_portfolio_allocation_chart(
                recommendations=recommendations,
                output_path="results/portfolio_allocation.png",
            )

        pdf_path = self.pdf_report_writer.build_report(
            summary_payload=summary_payload,
            channel_payloads=channel_streams,
            price_trends=price_trends,
            recommendations=recommendations,
            symbols=symbols,
            portfolio_recommendation=portfolio_rec,
            analysis_summary=reference_results.get("analysis_summary", {}),
            allocation_chart_path=allocation_chart_path,
            output_path=output_path,
        )

        return {"pdf_path": pdf_path, "summary": summary_payload}

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

    def print_multi_asset_plan(self, plan: Dict[str, Any]) -> None:
        budget = plan.get("budget", 0)
        print("\n" + "=" * 80)
        print("TRADEGRAPH MULTI-ASSET ALLOCATION PLAN")
        print("=" * 80)
        print(f"Budget: ${budget:,.2f}")

        strategies = plan.get("strategies", [])
        for strategy in strategies:
            print(
                f"\n📌 Strategy: {strategy.get('strategy', '').title()} - {strategy.get('description', '')}"
            )
            horizons = strategy.get("horizons", {})
            for horizon_key, payload in horizons.items():
                label = payload.get("label", horizon_key)
                print(f"  ➤ {label}: {payload.get('risk_focus', 'N/A')}")
                for allocation in payload.get("allocations", []):
                    percent = allocation.get("weight", 0) * 100
                    amount = allocation.get("amount", 0)
                    rationale = allocation.get("rationale", "")
                    sample_assets = ", ".join(
                        f"{asset['symbol']} ({asset['thesis']})"
                        for asset in allocation.get("sample_assets", [])
                    )
                    print(
                        f"    - {allocation.get('asset_class').upper()}: {percent:.1f}% "
                        f"(${amount:,.2f})"
                    )
                    if rationale:
                        print(f"      Rationale: {rationale}")
                    if sample_assets:
                        print(f"      Sample: {sample_assets}")

        notes = plan.get("notes") or []
        if notes:
            print("\n🗒 Advisor Notes:")
            for note in notes:
                print(f"  - {note}")
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
    parser.add_argument(
        "--channel-report",
        action="store_true",
        help="Generate the multichannel PDF report after analysis",
    )
    parser.add_argument(
        "--pdf-path",
        type=str,
        help="Optional output path for the PDF report",
    )
    parser.add_argument(
        "--multi-asset-budget",
        type=float,
        help="USD budget for a quick stocks/ETFs/crypto allocation plan",
    )
    parser.add_argument(
        "--multi-asset-strategies",
        type=str,
        help="Comma-separated strategies (growth,balanced,defensive,income)",
    )
    parser.add_argument(
        "--multi-asset-pdf-path",
        type=str,
        help="Optional output path for the multi-asset PDF report",
    )

    args = parser.parse_args()

    # Refresh the OpenAI API key so CLI runs pick up env changes immediately
    refresh_openai_api_key()

    # Configure logging
    logger.remove()  # Remove default handler
    logger.add(
        sys.stderr,
        level=settings.log_level,
        format="<green>{time}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
    )

    try:
        advisor = FinancialAdvisor()

        if args.multi_asset_budget:
            strategies = None
            if args.multi_asset_strategies:
                strategies = [
                    item.strip()
                    for item in args.multi_asset_strategies.split(",")
                    if item.strip()
                ]
            plan = await advisor.plan_multi_asset_allocation(
                budget=args.multi_asset_budget,
                strategies=strategies,
            )
            try:
                pdf_path = advisor.build_multi_asset_pdf(
                    plan, output_path=args.multi_asset_pdf_path
                )
                logger.info(f"Multi-asset PDF saved to: {pdf_path}")
                plan["pdf_path"] = pdf_path
            except Exception as pdf_exc:
                logger.warning(f"Failed to create multi-asset PDF: {pdf_exc}")
            if args.output_format == "json":
                import json

                print(json.dumps(plan, indent=2, default=str))
            else:
                advisor.print_multi_asset_plan(plan)
            return

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
                        output_path="results/portfolio_allocation.png",
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

            if args.channel_report:
                try:
                    existing_reference = (
                        results
                        if isinstance(results, dict) and results.get("channel_streams")
                        else None
                    )
                    pdf_info = await advisor.generate_channel_pdf_report(
                        symbols=args.symbols,
                        portfolio_size=args.portfolio_size,
                        risk_tolerance=args.risk_tolerance,
                        time_horizon=args.time_horizon,
                        include_reports=args.analysis_type == "comprehensive",
                        existing_results=existing_reference,
                        output_path=args.pdf_path,
                    )
                    logger.info(f"Channel PDF report saved to: {pdf_info['pdf_path']}")
                except Exception as pdf_exc:
                    logger.warning(f"Failed to create PDF channel report: {pdf_exc}")

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
