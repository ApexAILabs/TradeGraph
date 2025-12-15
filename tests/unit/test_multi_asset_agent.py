import pytest

from tradegraph_financial_advisor.agents.multi_asset_allocation_agent import (
    MultiAssetAllocationAgent,
)
from tradegraph_financial_advisor.reporting import MultiAssetPDFReportWriter


@pytest.mark.asyncio
async def test_multi_asset_agent_returns_balanced_plan():
    agent = MultiAssetAllocationAgent()
    result = await agent.execute({"budget": 10000, "strategies": ["growth", "unknown"]})

    assert result["budget"] == 10000.0
    assert result["strategies"], "Strategies should not be empty"
    plan = result["strategies"][0]
    assert plan["strategy"] == "growth"
    horizon = plan["horizons"]["1w"]
    weights = sum(item["weight"] for item in horizon["allocations"])
    amounts = sum(item["amount"] for item in horizon["allocations"])
    assert pytest.approx(weights, rel=1e-3) == 1.0
    assert pytest.approx(amounts, rel=1e-3) == 10000.0


def test_multi_asset_pdf_writer(tmp_path):
    writer = MultiAssetPDFReportWriter()
    plan = {
        "budget": 5000,
        "strategies": [
            {
                "strategy": "balanced",
                "description": "Balanced mix",
                "horizons": {
                    "1w": {
                        "label": "1-Week",
                        "risk_focus": "Liquidity",
                        "allocations": [
                            {
                                "asset_class": "stocks",
                                "weight": 0.5,
                                "amount": 2500,
                                "rationale": "Test rationale",
                                "sample_assets": [
                                    {"symbol": "AAPL", "thesis": "Quality"}
                                ],
                            }
                        ],
                    }
                },
            }
        ],
        "notes": ["Note"],
    }
    output_file = tmp_path / "multi_asset.pdf"
    pdf_path = writer.build_report(plan=plan, output_path=str(output_file))
    assert output_file.exists()
    assert pdf_path == str(output_file)
