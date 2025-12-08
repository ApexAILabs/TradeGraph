#!/usr/bin/env python3
"""
Advanced usage examples for TradeGraph Financial Advisor
"""

import asyncio
import json
from datetime import datetime

from tradegraph_financial_advisor import FinancialAdvisor
from tradegraph_financial_advisor.workflows.analysis_workflow import (
    FinancialAnalysisWorkflow,
)
from tradegraph_financial_advisor.agents.news_agent import NewsReaderAgent
from tradegraph_financial_advisor.agents.financial_agent import FinancialAnalysisAgent
from tradegraph_financial_advisor.agents.report_analysis_agent import (
    ReportAnalysisAgent,
)
from tradegraph_financial_advisor.agents.recommendation_engine import (
    TradingRecommendationEngine,
)
from tradegraph_financial_advisor.utils.helpers import (
    save_analysis_results,
    load_analysis_results,
    calculate_portfolio_metrics,
    PerformanceTimer,
)


async def individual_agents_example():
    """
    Example: Using individual agents directly for custom workflows
    """
    print("🔧 Using Individual Agents Example")
    print("=" * 50)

    # Initialize agents
    news_agent = NewsReaderAgent()
    financial_agent = FinancialAnalysisAgent()
    report_agent = ReportAnalysisAgent()

    symbols = ["AAPL", "MSFT"]

    try:
        # Start agents
        await news_agent.start()
        await financial_agent.start()
        await report_agent.start()

        # 1. Collect news data
        print("📰 Collecting news data...")
        news_input = {"symbols": symbols, "timeframe_hours": 48, "max_articles": 20}
        news_results = await news_agent.execute(news_input)
        print(f"Found {news_results.get('total_count', 0)} news articles")

        # 2. Get financial data
        print("💹 Analyzing financial data...")
        financial_input = {
            "symbols": symbols,
            "include_financials": True,
            "include_technical": True,
            "include_market_data": True,
        }
        financial_results = await financial_agent.execute(financial_input)
        print("Financial analysis completed")

        # 3. Analyze company reports
        print("📋 Analyzing company reports...")
        report_input = {
            "symbols": symbols,
            "report_types": ["10-K"],
            "analysis_depth": "standard",
        }
        report_results = await report_agent.execute(report_input)
        print("Report analysis completed")

        # Combine results
        combined_results = {
            "news_analysis": news_results,
            "financial_analysis": financial_results,
            "report_analysis": report_results,
            "timestamp": datetime.now().isoformat(),
        }

        return combined_results

    except Exception as e:
        print(f"❌ Individual agents example failed: {str(e)}")
        return None

    finally:
        # Always cleanup
        await news_agent.stop()
        await financial_agent.stop()
        await report_agent.stop()


async def custom_workflow_example():
    """
    Example: Creating a custom analysis workflow
    """
    print("🔄 Custom Workflow Example")
    print("=" * 50)

    # Create workflow components
    workflow = FinancialAnalysisWorkflow()
    recommendation_engine = TradingRecommendationEngine()

    symbols = ["TSLA", "NVDA"]

    try:
        # Run base workflow
        print("Running base financial analysis workflow...")
        portfolio_rec = await workflow.analyze_portfolio(
            symbols=symbols,
            portfolio_size=150000,
            risk_tolerance="aggressive",
            time_horizon="short_term",
        )

        # Enhance with custom recommendation logic
        print("Applying custom recommendation engine...")

        # Create analysis contexts (simplified)
        analysis_contexts = {}
        for symbol in symbols:
            analysis_contexts[symbol] = {
                "symbol": symbol,
                "market_data": {"current_price": 200.0},  # Placeholder
                "sentiment_analysis": {"sentiment_score": 0.1},  # Placeholder
            }

        # Generate enhanced recommendations
        await recommendation_engine.start()
        enhanced_results = await recommendation_engine.execute(
            {
                "analysis_contexts": analysis_contexts,
                "portfolio_constraints": {"portfolio_size": 150000, "max_positions": 5},
                "risk_preferences": {"risk_tolerance": "aggressive"},
            }
        )
        await recommendation_engine.stop()

        combined_results = {
            "base_workflow": portfolio_rec.dict() if portfolio_rec else None,
            "enhanced_recommendations": enhanced_results,
            "custom_workflow_metadata": {
                "workflow_type": "custom_enhanced",
                "enhancement_features": [
                    "custom_risk_analysis",
                    "enhanced_position_sizing",
                ],
            },
        }

        return combined_results

    except Exception as e:
        print(f"❌ Custom workflow example failed: {str(e)}")
        return None


async def portfolio_optimization_example():
    """
    Example: Advanced portfolio optimization with constraints
    """
    print("📊 Portfolio Optimization Example")
    print("=" * 50)

    advisor = FinancialAdvisor()

    # Define multiple portfolio scenarios
    scenarios = {
        "growth_portfolio": {
            "symbols": ["TSLA", "NVDA", "AMD", "SQ", "PLTR"],
            "portfolio_size": 100000,
            "risk_tolerance": "aggressive",
            "time_horizon": "medium_term",
        },
        "value_portfolio": {
            "symbols": ["BRK-B", "JPM", "JNJ", "PG", "WMT"],
            "portfolio_size": 100000,
            "risk_tolerance": "conservative",
            "time_horizon": "long_term",
        },
        "tech_portfolio": {
            "symbols": ["AAPL", "MSFT", "GOOGL", "AMZN", "META"],
            "portfolio_size": 100000,
            "risk_tolerance": "medium",
            "time_horizon": "medium_term",
        },
    }

    optimization_results = {}

    for scenario_name, config in scenarios.items():
        print(f"\n🔍 Optimizing {scenario_name}...")

        try:
            with PerformanceTimer(f"{scenario_name} optimization"):
                results = await advisor.analyze_portfolio(**config)

                # Calculate additional metrics
                portfolio_rec = results.get("portfolio_recommendation")
                if portfolio_rec and portfolio_rec.get("recommendations"):
                    metrics = calculate_portfolio_metrics(
                        portfolio_rec["recommendations"]
                    )
                    results["portfolio_metrics"] = metrics

                optimization_results[scenario_name] = results

                # Print summary
                if portfolio_rec:
                    print(
                        f"  ✅ {scenario_name}: {len(portfolio_rec['recommendations'])} recommendations"
                    )
                    print(
                        f"     Confidence: {portfolio_rec.get('total_confidence', 0):.1%}"
                    )
                    print(
                        f"     Risk Level: {portfolio_rec.get('overall_risk_level', 'unknown')}"
                    )

        except Exception as e:
            print(f"  ❌ {scenario_name} optimization failed: {str(e)}")

    # Compare portfolios
    print("\n📈 Portfolio Comparison:")
    for scenario_name, results in optimization_results.items():
        portfolio_rec = results.get("portfolio_recommendation", {})
        confidence = portfolio_rec.get("total_confidence", 0)
        risk = portfolio_rec.get("overall_risk_level", "unknown")
        print(
            f"  {scenario_name:20} | Confidence: {confidence:>6.1%} | Risk: {risk:>12}"
        )

    return optimization_results


async def real_time_monitoring_example():
    """
    Example: Real-time portfolio monitoring and alerts
    """
    print("⏰ Real-time Monitoring Example")
    print("=" * 50)

    advisor = FinancialAdvisor()

    # Portfolio to monitor
    portfolio_symbols = ["AAPL", "MSFT", "TSLA", "NVDA"]

    monitoring_results = {
        "monitoring_start": datetime.now().isoformat(),
        "portfolio_symbols": portfolio_symbols,
        "monitoring_cycles": [],
    }

    print(f"Monitoring portfolio: {', '.join(portfolio_symbols)}")

    # Simulate multiple monitoring cycles
    for cycle in range(3):  # 3 monitoring cycles
        print(f"\n📡 Monitoring Cycle {cycle + 1}/3")

        cycle_start = datetime.now()

        try:
            # Get current alerts
            alerts = await advisor.get_stock_alerts(portfolio_symbols)

            # Quick analysis for current state
            quick_analysis = await advisor.quick_analysis(
                symbols=portfolio_symbols, analysis_type="basic"
            )

            cycle_data = {
                "cycle_number": cycle + 1,
                "timestamp": cycle_start.isoformat(),
                "alerts": alerts,
                "quick_analysis": quick_analysis,
                "cycle_duration": (datetime.now() - cycle_start).total_seconds(),
            }

            monitoring_results["monitoring_cycles"].append(cycle_data)

            # Print cycle summary
            print(f"  Alerts generated: {len(alerts)}")
            if quick_analysis.get("recommendations"):
                buy_signals = sum(
                    1
                    for rec in quick_analysis["recommendations"]
                    if rec.get("recommendation") in ["buy", "strong_buy"]
                )
                print(f"  Buy signals: {buy_signals}/{len(portfolio_symbols)}")

        except Exception as e:
            print(f"  ❌ Monitoring cycle {cycle + 1} failed: {str(e)}")

        # Wait between cycles (in real scenario, this might be minutes/hours)
        if cycle < 2:  # Don't wait after the last cycle
            print("  ⏳ Waiting for next cycle...")
            await asyncio.sleep(2)  # Short wait for demo

    monitoring_results["monitoring_end"] = datetime.now().isoformat()
    monitoring_results["total_duration"] = (
        datetime.fromisoformat(monitoring_results["monitoring_end"])
        - datetime.fromisoformat(monitoring_results["monitoring_start"])
    ).total_seconds()

    print(
        f"\n✅ Monitoring completed in {monitoring_results['total_duration']:.1f} seconds"
    )

    return monitoring_results


async def data_export_import_example():
    """
    Example: Exporting and importing analysis results
    """
    print("💾 Data Export/Import Example")
    print("=" * 50)

    advisor = FinancialAdvisor()

    symbols = ["AAPL", "GOOGL"]

    try:
        # 1. Generate analysis
        print("Generating analysis for export...")
        results = await advisor.analyze_portfolio(
            symbols=symbols, portfolio_size=75000, risk_tolerance="medium"
        )

        # 2. Save results to file
        print("Saving results to file...")
        saved_file = save_analysis_results(results)
        print(f"Results saved to: {saved_file}")

        # 3. Load results from file
        print("Loading results from file...")
        loaded_results = load_analysis_results(saved_file)

        # 4. Verify data integrity
        original_timestamp = results.get("analysis_summary", {}).get(
            "analysis_timestamp"
        )
        loaded_timestamp = loaded_results.get("analysis_summary", {}).get(
            "analysis_timestamp"
        )

        if original_timestamp == loaded_timestamp:
            print("✅ Data integrity verified - timestamps match")
        else:
            print("❌ Data integrity issue - timestamps don't match")

        # 5. Export to custom format
        print("Exporting to custom format...")
        custom_export = {
            "export_metadata": {
                "export_time": datetime.now().isoformat(),
                "export_format": "tradegraph_v1",
                "symbols": symbols,
            },
            "recommendations_summary": [],
            "portfolio_metrics": {},
        }

        # Extract key information
        portfolio_rec = loaded_results.get("portfolio_recommendation", {})
        if portfolio_rec.get("recommendations"):
            for rec in portfolio_rec["recommendations"]:
                custom_export["recommendations_summary"].append(
                    {
                        "symbol": rec.get("symbol"),
                        "action": rec.get("recommendation"),
                        "confidence": rec.get("confidence_score"),
                        "allocation": rec.get("recommended_allocation"),
                    }
                )

        custom_export["portfolio_metrics"] = calculate_portfolio_metrics(
            portfolio_rec.get("recommendations", [])
        )

        # Save custom export
        custom_filename = (
            f"custom_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )
        with open(f"results/{custom_filename}", "w") as f:
            json.dump(custom_export, f, indent=2, default=str)

        print(f"Custom export saved to: results/{custom_filename}")

        return {
            "original_results": results,
            "loaded_results": loaded_results,
            "custom_export": custom_export,
            "files_created": [saved_file, f"results/{custom_filename}"],
        }

    except Exception as e:
        print(f"❌ Data export/import example failed: {str(e)}")
        return None


async def error_handling_example():
    """
    Example: Robust error handling and recovery
    """
    print("🛡️ Error Handling and Recovery Example")
    print("=" * 50)

    advisor = FinancialAdvisor()

    # Test scenarios with potential errors
    test_scenarios = [
        {
            "name": "Invalid Symbols",
            "symbols": ["INVALID", "BADTICKER", "NOTREAL"],
            "should_fail": True,
        },
        {"name": "Valid Symbols", "symbols": ["AAPL", "MSFT"], "should_fail": False},
        {
            "name": "Mixed Valid/Invalid",
            "symbols": ["AAPL", "INVALID", "MSFT"],
            "should_fail": False,  # Should handle gracefully
        },
    ]

    error_handling_results = {}

    for scenario in test_scenarios:
        scenario_name = scenario["name"]
        symbols = scenario["symbols"]
        should_fail = scenario.get("should_fail", False)

        print(f"\n🧪 Testing: {scenario_name}")
        print(f"   Symbols: {symbols}")

        try:
            # Try quick analysis (more likely to succeed even with some invalid symbols)
            results = await advisor.quick_analysis(
                symbols=symbols, analysis_type="basic"
            )

            if should_fail:
                print("   ⚠️ Expected failure but analysis succeeded")
                status = "unexpected_success"
            else:
                print("   ✅ Analysis completed successfully")
                status = "success"

            error_handling_results[scenario_name] = {
                "status": status,
                "results": results,
                "error": None,
            }

        except Exception as e:
            if should_fail:
                print(f"   ✅ Expected failure occurred: {str(e)[:100]}...")
                status = "expected_failure"
            else:
                print(f"   ❌ Unexpected failure: {str(e)[:100]}...")
                status = "unexpected_failure"

            error_handling_results[scenario_name] = {
                "status": status,
                "results": None,
                "error": str(e),
            }

    # Summary
    print("\n📊 Error Handling Summary:")
    for scenario_name, result in error_handling_results.items():
        status = result["status"]
        print(f"   {scenario_name:20} | Status: {status}")

    return error_handling_results


async def main():
    """
    Run all advanced examples
    """
    print("🚀 TradeGraph Financial Advisor - Advanced Examples")
    print("=" * 70)

    # Advanced examples
    examples = [
        ("Individual Agents Usage", individual_agents_example),
        ("Custom Workflow Creation", custom_workflow_example),
        ("Portfolio Optimization", portfolio_optimization_example),
        ("Real-time Monitoring", real_time_monitoring_example),
        ("Data Export/Import", data_export_import_example),
        ("Error Handling & Recovery", error_handling_example),
    ]

    all_results = {}

    for example_name, example_func in examples:
        try:
            print(f"\n{'=' * 70}")
            with PerformanceTimer(example_name):
                result = await example_func()
                all_results[example_name] = result

            print(f"✅ {example_name} completed")

        except KeyboardInterrupt:
            print(f"\n⏹️ {example_name} interrupted by user")
            break

        except Exception as e:
            print(f"\n❌ {example_name} failed: {str(e)}")
            all_results[example_name] = {"error": str(e)}

        # Brief pause between examples
        await asyncio.sleep(1)

    print(f"\n{'=' * 70}")
    print("🎉 All advanced examples completed!")

    # Save comprehensive results
    try:
        comprehensive_results = {
            "session_metadata": {
                "session_start": datetime.now().isoformat(),
                "examples_run": list(all_results.keys()),
                "total_examples": len(examples),
            },
            "example_results": all_results,
        }

        results_file = save_analysis_results(
            comprehensive_results,
            f"advanced_examples_session_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
        )
        print(f"📁 Comprehensive results saved to: {results_file}")

    except Exception as e:
        print(f"⚠️ Could not save comprehensive results: {str(e)}")

    print("\n🎓 Advanced Examples Complete!")
    print("These examples demonstrate:")
    print("- Using individual agents for custom workflows")
    print("- Portfolio optimization strategies")
    print("- Real-time monitoring capabilities")
    print("- Data persistence and export features")
    print("- Robust error handling patterns")


if __name__ == "__main__":
    asyncio.run(main())
