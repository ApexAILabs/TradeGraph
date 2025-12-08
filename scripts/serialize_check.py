import asyncio
import json
from pprint import pprint

from tradegraph_financial_advisor.main import FinancialAdvisor

async def main():
    advisor = FinancialAdvisor()
    result = await advisor.quick_analysis(["AAPL"], analysis_type="detailed")
    print("Analysis result type:", type(result))
    json.dumps(result, ensure_ascii=False, default=str)
    print("Serialization succeeded")
    pprint(result.get("database_insights", {}).get("recent_queries"))

if __name__ == "__main__":
    asyncio.run(main())
