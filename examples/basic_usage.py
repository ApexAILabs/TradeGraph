#!/usr/bin/env python3
"""
Basic usage example for RADGEGRAPH Financial Advisor
"""

import asyncio
import os
from radgegraph_financial_advisor import FinancialAdvisor


async def basic_analysis_example():
    """
    Example: Basic portfolio analysis for a few stocks
    """
    print("🔍 Running Basic Analysis Example")
    print("=" * 50)

    # Initialize the advisor
    advisor = FinancialAdvisor()

    # Define stocks to analyze
    symbols = ["AAPL", "MSFT", "GOOGL"]

    try:
        # Run standard analysis
        results = await advisor.analyze_portfolio(
            symbols=symbols,
            portfolio_size=100000,  # $100k portfolio
            risk_tolerance="medium",
            time_horizon="medium_term",
            include_reports=False,  # Skip SEC filings for faster analysis
        )

        # Print results to console
        advisor.print_recommendations(results)

        # You can also access individual components:
        portfolio_rec = results.get("portfolio_recommendation")
        if portfolio_rec:
            print("\n📊 Portfolio Summary:")
            print(f"Total Confidence: {portfolio_rec['total_confidence']:.1%}")
            print(f"Number of Recommendations: {len(portfolio_rec['recommendations'])}")

        return results

    except Exception as e:
        print(f"❌ Analysis failed: {str(e)}")
        print("Make sure you have set up your API keys in the .env file")
        return None


async def quick_analysis_example():
    """
    Example: Quick analysis for rapid insights
    """
    print("\n⚡ Running Quick Analysis Example")
    print("=" * 50)

    advisor = FinancialAdvisor()

    # Quick analysis for tech stocks
    symbols = ["TSLA", "NVDA", "AMD"]

    try:
        results = await advisor.quick_analysis(symbols=symbols, analysis_type="basic")

        print(f"Quick analysis completed for: {', '.join(symbols)}")

        if "recommendations" in results:
            for rec in results["recommendations"]:
                symbol = rec.get("symbol", "Unknown")
                recommendation = rec.get("recommendation", "hold").upper()
                confidence = rec.get("confidence_score", 0)
                print(f"{symbol}: {recommendation} (Confidence: {confidence:.1%})")

        return results

    except Exception as e:
        print(f"❌ Quick analysis failed: {str(e)}")
        return None


async def alerts_example():
    """
    Example: Generate trading alerts
    """
    print("\n🚨 Running Alerts Example")
    print("=" * 50)

    advisor = FinancialAdvisor()

    # Monitor these stocks for alerts
    symbols = ["AAPL", "TSLA", "NVDA"]

    try:
        alerts = await advisor.get_stock_alerts(symbols)

        print(f"Generated {len(alerts)} alerts:")
        for alert in alerts:
            symbol = alert.get("symbol", "Unknown")
            message = alert.get("message", "No message")
            urgency = alert.get("urgency", "low")
            print(f"  {urgency.upper()}: {symbol} - {message}")

        return alerts

    except Exception as e:
        print(f"❌ Alert generation failed: {str(e)}")
        return None


async def portfolio_comparison_example():
    """
    Example: Compare different portfolio configurations
    """
    print("\n📈 Running Portfolio Comparison Example")
    print("=" * 50)

    advisor = FinancialAdvisor()

    # Define different portfolios to compare
    portfolios = {
        "Tech Focus": ["AAPL", "MSFT", "GOOGL", "NVDA"],
        "Diversified": ["AAPL", "JPM", "JNJ", "XOM"],
        "Growth": ["TSLA", "NVDA", "AMD", "SQ"],
    }

    results = {}

    for portfolio_name, symbols in portfolios.items():
        print(f"\nAnalyzing {portfolio_name} portfolio...")

        try:
            result = await advisor.quick_analysis(
                symbols=symbols, analysis_type="standard"
            )

            results[portfolio_name] = result

            # Quick summary
            if "recommendations" in result:
                buy_count = sum(
                    1
                    for rec in result["recommendations"]
                    if rec.get("recommendation") in ["buy", "strong_buy"]
                )
                print(
                    f"  {portfolio_name}: {buy_count}/{len(symbols)} BUY recommendations"
                )

        except Exception as e:
            print(f"  ❌ {portfolio_name} analysis failed: {str(e)}")

    return results


async def custom_parameters_example():
    """
    Example: Using custom analysis parameters
    """
    print("\n⚙️ Running Custom Parameters Example")
    print("=" * 50)

    advisor = FinancialAdvisor()

    # Aggressive high-risk portfolio
    aggressive_config = {
        "symbols": ["TSLA", "PLTR", "ARKK"],
        "portfolio_size": 50000,
        "risk_tolerance": "aggressive",
        "time_horizon": "short_term",
    }

    # Conservative low-risk portfolio
    conservative_config = {
        "symbols": ["JNJ", "PG", "KO"],
        "portfolio_size": 200000,
        "risk_tolerance": "conservative",
        "time_horizon": "long_term",
    }

    configs = {"Aggressive": aggressive_config, "Conservative": conservative_config}

    for config_name, config in configs.items():
        print(f"\n{config_name} Portfolio Analysis:")

        try:
            results = await advisor.analyze_portfolio(**config)

            portfolio_rec = results.get("portfolio_recommendation")
            if portfolio_rec:
                risk_level = portfolio_rec.get("overall_risk_level", "unknown")
                confidence = portfolio_rec.get("total_confidence", 0)
                print(f"  Risk Level: {risk_level}")
                print(f"  Confidence: {confidence:.1%}")

        except Exception as e:
            print(f"  ❌ {config_name} analysis failed: {str(e)}")


def check_environment():
    """
    Check if required environment variables are set
    """
    required_vars = ["OPENAI_API_KEY"]

    missing_vars = []
    for var in required_vars:
        if not os.getenv(var):
            missing_vars.append(var)

    if missing_vars:
        print("❌ Missing required environment variables:")
        for var in missing_vars:
            print(f"   - {var}")
        print("\nPlease set these in your .env file or environment")
        return False

    print("✅ Environment variables are configured")
    return True


async def main():
    """
    Run all examples
    """
    print("🚀 RADGEGRAPH Financial Advisor - Examples")
    print("=" * 60)

    # Check environment first
    if not check_environment():
        print(
            "\n⚠️ Please configure your environment variables before running examples"
        )
        return

    # Run examples
    examples = [
        ("Basic Analysis", basic_analysis_example),
        ("Quick Analysis", quick_analysis_example),
        ("Trading Alerts", alerts_example),
        ("Portfolio Comparison", portfolio_comparison_example),
        ("Custom Parameters", custom_parameters_example),
    ]

    for example_name, example_func in examples:
        try:
            print(f"\n{'=' * 60}")
            await example_func()
            print(f"✅ {example_name} completed successfully")

        except KeyboardInterrupt:
            print(f"\n⏹️ {example_name} interrupted by user")
            break

        except Exception as e:
            print(f"\n❌ {example_name} failed: {str(e)}")

        # Small delay between examples
        await asyncio.sleep(1)

    print(f"\n{'=' * 60}")
    print("🎉 All examples completed!")
    print("\nNext steps:")
    print("- Try modifying the stock symbols in the examples")
    print("- Experiment with different risk tolerances and time horizons")
    print("- Check out the comprehensive analysis with include_reports=True")
    print("- Read the documentation for advanced usage patterns")


if __name__ == "__main__":
    asyncio.run(main())
