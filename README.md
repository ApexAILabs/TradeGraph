<p align="center">
  <img src="assets/logo.png" alt="TradeGraph Logo" width="200"/>
</p>

# TradeGraph Financial Advisor

A sophisticated multi-agent financial analysis system that uses **LangGraph**, **ddgs**, and **Crawl4AI** to provide intelligent trading recommendations based on real-time financial news and comprehensive company analysis.

## 🚀 Features

- **Multi-Agent Architecture**: Coordinated agents for news analysis, financial data processing, and report analysis
- **SEC Filing Analysis**: Deep analysis of 10-K and 10-Q reports using AI
- **Technical Analysis**: Comprehensive technical indicators and chart pattern recognition
- **Sentiment Analysis**: AI-powered sentiment analysis of news and social media
- **Portfolio Optimization**: Intelligent portfolio construction with risk management
- **Trading Recommendations**: Buy/Sell/Hold recommendations with confidence scores
- **Risk Assessment**: Multi-factor risk analysis and position sizing

## 📋 Requirements

- Python 3.10+
- OpenAI API key
- Optional: Alpha Vantage API key for enhanced financial data

## 🛠️ Installation

### Quick Installation

```bash
# Clone the repository
git clone https://github.com/Mehranmzn/TradeGraph.git
cd financial-advisor

# Install the package
pip install -e .

# Or install from PyPI (when published)
pip install tradegraph-financial-advisor
```

### Development Installation

```bash
# Clone and install in development mode
git clone https://github.com/Mehranmzn/TradeGraph.git
cd financial-advisor

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\\Scripts\\activate

# Install with development dependencies
pip install -e ".[dev]"
```

## ⚙️ Configuration

### Environment Variables

Create a `.env` file based on `.env.example`:

```bash
cp .env.example .env
```

Required environment variables:

```env
OPENAI_API_KEY=your_openai_api_key_here
# Optional but recommended
ALPHA_VANTAGE_API_KEY=your_alpha_vantage_api_key_here
FINANCIAL_DATA_API_KEY=your_financial_data_api_key_here

# Configuration
LOG_LEVEL=INFO
MAX_CONCURRENT_AGENTS=5
ANALYSIS_TIMEOUT_SECONDS=30
NEWS_SOURCES=bloomberg,reuters,yahoo-finance,marketwatch,cnbc
DEFAULT_PORTFOLIO_SIZE=100000
```

## 🎯 Quick Start

### Command Line Usage

```bash
# Basic analysis
tradegraph AAPL MSFT GOOGL

# Comprehensive analysis with custom parameters
tradegraph AAPL MSFT GOOGL \
  --portfolio-size 250000 \
  --risk-tolerance aggressive \
  --time-horizon long_term \
  --analysis-type comprehensive

# Quick analysis
tradegraph TSLA NVDA --analysis-type quick

# Generate alerts only
tradegraph AAPL --alerts-only

# JSON output
tradegraph AAPL MSFT --output-format json > analysis.json
```

### Python API Usage

```python
import asyncio
from tradegraph_financial_advisor import FinancialAdvisor

async def main():
    advisor = FinancialAdvisor()

    # Comprehensive analysis
    results = await advisor.analyze_portfolio(
        symbols=["AAPL", "MSFT", "GOOGL"],
        portfolio_size=100000,
        risk_tolerance="medium",
        time_horizon="medium_term",
        include_reports=True
    )

    # Print results
    advisor.print_recommendations(results)

    # Quick analysis
    quick_results = await advisor.quick_analysis(
        symbols=["TSLA", "NVDA"],
        analysis_type="standard"
    )

    # Generate alerts
    alerts = await advisor.get_stock_alerts(["AAPL", "MSFT"])

    return results

# Run the analysis
results = asyncio.run(main())
```

### Advanced Usage

```python
from tradegraph_financial_advisor.workflows import FinancialAnalysisWorkflow
from tradegraph_financial_advisor.agents import (
    NewsReaderAgent,
    FinancialAnalysisAgent,
    ReportAnalysisAgent
)

async def advanced_analysis():
    # Use individual agents
    news_agent = NewsReaderAgent()
    await news_agent.start()

    news_data = await news_agent.execute({
        "symbols": ["AAPL"],
        "timeframe_hours": 24,
        "max_articles": 50
    })

    await news_agent.stop()

    # Use the full workflow
    workflow = FinancialAnalysisWorkflow()
    portfolio_rec = await workflow.analyze_portfolio(
        symbols=["AAPL", "MSFT"],
        portfolio_size=500000,
        risk_tolerance="aggressive"
    )

    return portfolio_rec
```

## 📊 Output Examples

### Console Output

```
================================================================================
TRADEGRAPH FINANCIAL ADVISOR - ANALYSIS RESULTS
================================================================================

Analysis Date: 2024-12-20T15:30:00
Symbols Analyzed: AAPL, MSFT, GOOGL
Portfolio Size: $100,000.00
Risk Tolerance: medium

📊 PORTFOLIO RECOMMENDATION
Overall Confidence: 75.2%
Diversification Score: 80.0%
Risk Level: medium

📈 INDIVIDUAL RECOMMENDATIONS (3 stocks):
------------------------------------------------------------

AAPL: BUY (Confidence: 78.5%)
  Current: $195.89 | Target: $225.00
  Allocation: 8.5% | Risk: medium
  Key Factors: Strong iPhone 15 sales, Services growth

MSFT: STRONG_BUY (Confidence: 85.2%)
  Current: $374.50 | Target: $420.00
  Allocation: 12.0% | Risk: low
  Key Factors: Azure growth, AI integration

GOOGL: HOLD (Confidence: 65.0%)
  Current: $140.25 | Target: $150.00
  Allocation: 5.5% | Risk: medium
  Key Factors: Search dominance, Cloud recovery
```

### JSON Output Structure

```json
{
  "analysis_summary": {
    "symbols_analyzed": ["AAPL", "MSFT"],
    "portfolio_size": 100000,
    "risk_tolerance": "medium",
    "analysis_timestamp": "2024-12-20T15:30:00"
  },
  "portfolio_recommendation": {
    "recommendations": [
      {
        "symbol": "AAPL",
        "recommendation": "buy",
        "confidence_score": 0.785,
        "target_price": 225.00,
        "current_price": 195.89,
        "recommended_allocation": 0.085,
        "risk_level": "medium",
        "key_factors": ["Strong iPhone 15 sales", "Services growth"],
        "analyst_notes": "Strong fundamentals with growth catalysts"
      }
    ],
    "total_confidence": 0.752,
    "diversification_score": 0.80,
    "overall_risk_level": "medium"
  },
  "detailed_reports": {
    "AAPL": {
      "financial_health_score": 8.5,
      "executive_summary": "Apple maintains strong financial position...",
      "key_metrics": {...},
      "risk_factors": [...],
      "growth_prospects": [...]
    }
  }
}
```

## 🏗️ Architecture

### Multi-Agent System

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   News Reader   │    │ Financial Agent  │    │ Report Analyzer │
│     Agent       │    │                  │    │     Agent       │
├─────────────────┤    ├──────────────────┤    ├─────────────────┤
│ • Web scraping  │    │ • Market data    │    │ • SEC filings   │
│ • News analysis │    │ • Technical      │    │ • 10-K/10-Q     │
│ • Sentiment     │    │   indicators     │    │ • AI analysis   │
└─────────────────┘    └──────────────────┘    └─────────────────┘
         │                       │                       │
         └───────────────────────┼───────────────────────┘
                                 │
                    ┌─────────────────────┐
                    │   LangGraph        │
                    │   Workflow         │
                    │   Coordinator      │
                    └─────────────────────┘
                                 │
                    ┌─────────────────────┐
                    │  Recommendation    │
                    │     Engine         │
                    └─────────────────────┘
```

### Technology Stack

- **LangGraph**: Workflow orchestration and agent coordination
- **OpenAI GPT-4**: Natural language processing and analysis
- **yfinance**: Financial data retrieval
- **pandas/numpy**: Data processing and analysis
- **aiohttp**: Async HTTP requests
- **pydantic**: Data validation and serialization

## 📈 Supported Analysis Types

### 1. Quick Analysis
- Basic market data
- Simple news sentiment
- Fast recommendations
- ~30 seconds execution time

### 2. Standard Analysis
- Comprehensive market data
- Technical indicators
- News analysis
- Basic portfolio optimization
- ~2-3 minutes execution time

### 3. Comprehensive Analysis
- Everything in Standard
- SEC filing analysis
- Deep fundamental analysis
- Advanced portfolio optimization
- Risk correlation analysis
- ~5-10 minutes execution time

## 🔧 Customization

### Adding Custom Agents

```python
from tradegraph_financial_advisor.agents import BaseAgent

class CustomAnalysisAgent(BaseAgent):
    def __init__(self, **kwargs):
        super().__init__(
            name="CustomAnalysisAgent",
            description="Custom analysis functionality",
            **kwargs
        )

    async def execute(self, input_data):
        # Your custom analysis logic
        return {"analysis_result": "custom_data"}
```

### Custom Workflows

```python
from langgraph.graph import StateGraph
from tradegraph_financial_advisor.workflows import AnalysisState

def create_custom_workflow():
    workflow = StateGraph(AnalysisState)

    # Add your custom nodes
    workflow.add_node("custom_analysis", custom_analysis_node)

    # Define workflow
    workflow.set_entry_point("custom_analysis")
    workflow.add_edge("custom_analysis", END)

    return workflow.compile()
```

## 🧪 Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=tradegraph_financial_advisor

# Run specific test categories
pytest tests/test_agents.py
pytest tests/test_workflows.py
```

## 📝 Development

### Code Quality

```bash
# Format code
black src/
isort src/

# Lint code
flake8 src/

# Type checking
mypy src/
```

### Pre-commit Hooks

```bash
# Install pre-commit hooks
pre-commit install

# Run hooks manually
pre-commit run --all-files
```

## 🚨 Important Notes

### Rate Limits
- **OpenAI API**: Monitor usage to avoid rate limits
- **Financial APIs**: Most have daily/monthly limits

### Data Accuracy
- Market data may have delays (typically 15-20 minutes)
- News sentiment is AI-generated and should be verified
- SEC filing analysis is automated and may miss nuances
- **Always conduct your own research before making investment decisions**

### Legal Disclaimer
This software is for educational and research purposes only. It does not constitute financial advice. Always consult with qualified financial advisors before making investment decisions.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🆘 Support

- **GitHub Issues**: [Report bugs or request features](https://github.com/tradegraph/financial-advisor/issues)
- **Documentation**: [Full documentation](https://deepwiki.com/Mehranmzn/TradeGraph)
- **Community**: [Discord server](https://discord.gg/tradegraph)

## 🙏 Acknowledgments

- **LangGraph** team for the excellent multi-agent framework
- **OpenAI** for powerful language models
- **Financial data providers** for market data access

---

**⚠️ Investment Disclaimer**: This tool provides analysis and suggestions based on available data and AI models. It is not a substitute for professional financial advice. Always do your own research and consider consulting with financial advisors before making investment decisions. Past performance does not guarantee future results.
