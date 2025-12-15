import os
import pandas as pd
import pytest

from tradegraph_financial_advisor.services.channel_stream_service import (
    FinancialNewsChannelService,
    ChannelType,
)
from tradegraph_financial_advisor.services.price_trend_service import PriceTrendService
from tradegraph_financial_advisor.agents.channel_report_agent import ChannelReportAgent
from tradegraph_financial_advisor.reporting import ChannelPDFReportWriter


class _DummyPriceService:
    def __init__(self, payload):
        self.payload = payload

    async def get_trends_for_symbols(self, symbols):
        return {symbol: self.payload[next(iter(self.payload))] for symbol in symbols}


@pytest.mark.asyncio
async def test_channel_service_price_payload(sample_price_trends):
    service = FinancialNewsChannelService(
        price_service=_DummyPriceService(sample_price_trends)
    )
    payload = await service.fetch_channel_payload(
        ChannelType.LIVE_PRICE_STREAM.value, symbols=["AAPL"]
    )
    assert payload["items"][0]["symbol"] == "AAPL"
    assert "trends" in payload["items"][0]


@pytest.mark.asyncio
async def test_channel_report_agent_fallback(
    sample_channel_streams, sample_price_trends, sample_recommendations
):
    agent = ChannelReportAgent(llm_client=None, enable_llm=False)
    summary = await agent.execute(
        {
            "channel_payloads": sample_channel_streams,
            "price_trends": sample_price_trends,
            "recommendations": sample_recommendations,
        }
    )
    assert summary["news_takeaways"]
    assert "summary_text" in summary
    assert summary.get("advisor_memo")
    assert summary.get("guidance_points")
    assert summary.get("key_stats", {}).get("channel_count") == 1


def test_pdf_report_writer(
    tmp_path, sample_channel_streams, sample_price_trends, sample_recommendations
):
    writer = ChannelPDFReportWriter()
    output_file = tmp_path / "report.pdf"
    summary_payload = {
        "summary_text": "Markets steady amid mixed data.",
        "news_takeaways": ["Headline one", "Headline two"],
        "risk_assessment": "medium",
        "buy_or_sell_view": "buy",
        "trend_commentary": "AAPL: +5% YoY",
        "advisor_memo": "Maintain constructive stance with risk controls.",
        "price_action_notes": ["AAPL: +2.5% weekly"],
        "guidance_points": ["Add MSFT on earnings strength"],
        "key_stats": {
            "channel_count": 1,
            "headline_count": 2,
            "recommendation_count": 2,
        },
    }
    pdf_path = writer.build_report(
        summary_payload=summary_payload,
        channel_payloads=sample_channel_streams,
        price_trends=sample_price_trends,
        recommendations=sample_recommendations,
        symbols=["AAPL"],
        analysis_summary={
            "portfolio_size": 100000,
            "risk_tolerance": "medium",
            "time_horizon": "medium_term",
            "symbols_analyzed": ["AAPL"],
        },
        portfolio_recommendation={
            "total_confidence": 0.8,
            "diversification_score": 0.7,
            "expected_return": 0.12,
            "expected_volatility": 0.2,
            "overall_risk_level": "medium",
        },
        output_path=str(output_file),
    )
    assert os.path.exists(pdf_path)
    assert os.path.getsize(pdf_path) > 0


def test_price_trend_service_summarize_series():
    series = pd.Series([100.0, 105.0, 110.0])
    summary = PriceTrendService._summarize_series(series)
    assert summary["direction"] == "bullish"
    assert pytest.approx(summary["percent_change"], rel=1e-3) == 10.0
