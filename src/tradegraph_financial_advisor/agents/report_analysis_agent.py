import json
from typing import Any, Dict, List, Optional
from datetime import datetime
from loguru import logger
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

from .base_agent import BaseAgent
from ..services.local_scraping_service import LocalScrapingService
from ..config.settings import settings


class ReportAnalysisAgent(BaseAgent):
    def __init__(
        self,
        scraping_service: Optional[LocalScrapingService] = None,
        llm_model_name: str = "gpt-5-nano",
        **kwargs,
    ):
        super().__init__(
            name="ReportAnalysisAgent",
            description="Analyzes company financial reports and SEC filings for deep fundamental analysis",
            **kwargs,
        )
        self.local_scraping_service = scraping_service or LocalScrapingService()
        self.llm_model_name = llm_model_name
        self.llm = ChatOpenAI(
            model=self.llm_model_name, temperature=0.1, api_key=settings.openai_api_key
        )

    async def start(self) -> None:
        await super().start()
        await self.local_scraping_service.start()

    async def stop(self) -> None:
        await self.local_scraping_service.stop()
        await super().stop()

    async def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        symbols = input_data.get("symbols", [])
        report_types = input_data.get("report_types", ["10-K", "10-Q"])
        analysis_depth = input_data.get("analysis_depth", "detailed")

        logger.info(f"Analyzing company reports for symbols: {symbols}")

        results = {}

        for symbol in symbols:
            try:
                symbol_analysis = await self._analyze_company_reports(
                    symbol, report_types, analysis_depth
                )
                results[symbol] = symbol_analysis

            except Exception as e:
                logger.error(f"Error analyzing reports for {symbol}: {str(e)}")
                results[symbol] = {"error": str(e)}

        return {
            "report_analysis": results,
            "analysis_timestamp": datetime.now().isoformat(),
        }

    async def _analyze_company_reports(
        self, symbol: str, report_types: List[str], analysis_depth: str
    ) -> Dict[str, Any]:
        try:
            analysis_results = {
                "symbol": symbol,
                "report_analyses": [],
                "executive_summary": "",
                "key_metrics": {},
                "risk_factors": [],
                "growth_prospects": [],
                "competitive_position": "",
                "financial_health_score": 0.0,
            }

            # Scrape and analyze each report type
            for report_type in report_types:
                try:
                    filings = await self.local_scraping_service.search_and_scrape_financial_reports(
                        symbol, report_type
                    )

                    for filing in filings[:2]:  # Analyze 2 most recent filings
                        report_analysis = await self._analyze_single_report(
                            symbol, filing, report_type, analysis_depth
                        )
                        analysis_results["report_analyses"].append(report_analysis)

                except Exception as e:
                    logger.warning(
                        f"Failed to analyze {report_type} for {symbol}: {str(e)}"
                    )

            # Generate comprehensive analysis summary
            if analysis_results["report_analyses"]:
                summary = await self._generate_comprehensive_summary(
                    symbol, analysis_results["report_analyses"]
                )
                analysis_results.update(summary)

            return analysis_results

        except Exception as e:
            logger.error(f"Error in comprehensive analysis for {symbol}: {str(e)}")
            return {"error": str(e)}

    async def _analyze_single_report(
        self, symbol: str, filing: Dict[str, Any], report_type: str, analysis_depth: str
    ) -> Dict[str, Any]:
        try:
            content = filing.get("content", "")

            if analysis_depth == "detailed":
                analysis_prompt = f"""
                Perform a detailed analysis of this {report_type} filing for {symbol}.
                Extract and analyze the following information:

                Filing Content (first 8000 characters):
                {content[:8000]}

                Please provide analysis in JSON format with these sections:

                {{
                    "financial_highlights": {{
                        "revenue": "amount and growth",
                        "net_income": "amount and growth",
                        "eps": "earnings per share",
                        "cash_flow": "operating cash flow",
                        "debt_levels": "total debt and debt-to-equity",
                        "return_metrics": "ROE, ROA if available"
                    }},
                    "business_developments": [
                        "key business updates, expansions, new products, etc."
                    ],
                    "risk_factors": [
                        "specific risks mentioned in the filing"
                    ],
                    "management_outlook": "forward-looking statements and guidance",
                    "competitive_dynamics": "market position and competition mentions",
                    "capital_allocation": "dividends, buybacks, capex plans",
                    "segment_performance": "performance by business segment if available",
                    "key_metrics": {{
                        "gross_margin": "percentage if available",
                        "operating_margin": "percentage if available",
                        "asset_turnover": "efficiency metrics if available"
                    }},
                    "red_flags": [
                        "any concerning items like accounting changes, going concern, etc."
                    ],
                    "growth_drivers": [
                        "factors that could drive future growth"
                    ],
                    "overall_assessment": "summary assessment of company health and prospects"
                }}
                """
            else:
                analysis_prompt = f"""
                Provide a concise analysis of this {report_type} filing for {symbol}.

                Filing Content (first 4000 characters):
                {content[:4000]}

                Provide analysis in JSON format:
                {{
                    "key_financials": "revenue, profit, key metrics",
                    "main_developments": "important business updates",
                    "risks": ["top 3 risks"],
                    "outlook": "management guidance and prospects",
                    "assessment": "overall health score 1-10 and brief reasoning"
                }}
                """

            response = await self.llm.ainvoke([HumanMessage(content=analysis_prompt)])

            try:
                analysis_data = json.loads(response.content)
                analysis_data["report_type"] = report_type
                analysis_data["filing_url"] = filing.get("url", "")
                analysis_data["analysis_date"] = datetime.now().isoformat()
                return analysis_data

            except json.JSONDecodeError as e:
                logger.warning(f"Failed to parse LLM response for {symbol}: {str(e)}")
                return {
                    "error": "Failed to parse analysis",
                    "raw_response": response.content[:500],
                    "report_type": report_type,
                }

        except Exception as e:
            logger.error(f"Error analyzing single report for {symbol}: {str(e)}")
            return {"error": str(e), "report_type": report_type}

    async def _generate_comprehensive_summary(
        self, symbol: str, report_analyses: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        try:
            # Combine all analyses for comprehensive view
            combined_analysis_prompt = f"""
            Based on the following financial report analyses for {symbol},
            provide a comprehensive investment summary:

            Report Analyses:
            {json.dumps(report_analyses, indent=2, default=str)}

            Generate a comprehensive summary in JSON format:

            {{
                "executive_summary": "2-3 paragraph summary of company's current state and prospects",
                "key_metrics": {{
                    "financial_strength": "score 1-10 with reasoning",
                    "growth_potential": "score 1-10 with reasoning",
                    "competitive_position": "score 1-10 with reasoning",
                    "management_quality": "score 1-10 with reasoning"
                }},
                "risk_factors": [
                    "top 5 most significant risks based on all reports"
                ],
                "growth_prospects": [
                    "top 5 growth drivers and opportunities"
                ],
                "competitive_position": "detailed assessment of market position",
                "financial_health_score": 0.0-10.0,
                "investment_thesis": {{
                    "bull_case": "strongest arguments for investing",
                    "bear_case": "strongest arguments against investing",
                    "key_catalysts": ["events that could drive stock performance"],
                    "key_risks": ["events that could hurt stock performance"]
                }},
                "peer_comparison_notes": "how company compares to industry peers",
                "valuation_commentary": "thoughts on current valuation based on fundamentals"
            }}
            """

            response = await self.llm.ainvoke(
                [HumanMessage(content=combined_analysis_prompt)]
            )

            try:
                summary_data = json.loads(response.content)
                return summary_data

            except json.JSONDecodeError as e:
                logger.warning(
                    f"Failed to parse comprehensive summary for {symbol}: {str(e)}"
                )
                return {
                    "executive_summary": "Analysis completed but summary parsing failed",
                    "financial_health_score": 5.0,
                    "key_metrics": {},
                    "risk_factors": [],
                    "growth_prospects": [],
                }

        except Exception as e:
            logger.error(
                f"Error generating comprehensive summary for {symbol}: {str(e)}"
            )
            return {
                "error": str(e),
                "executive_summary": "Failed to generate summary",
                "financial_health_score": 0.0,
            }

    async def analyze_earnings_calls(
        self, symbol: str, quarters: int = 4
    ) -> Dict[str, Any]:
        try:
            # Search for earnings call transcripts
            search_queries = [
                f"{symbol} earnings call transcript Q1 2024",
                f"{symbol} earnings call transcript Q2 2024",
                f"{symbol} earnings call transcript Q3 2024",
                f"{symbol} earnings call transcript Q4 2024",
            ]

            earnings_analyses = []

            for query in search_queries[:quarters]:
                try:
                    # Use web search or specific financial sites
                    # This would typically use a more sophisticated search
                    # For demo purposes, showing the structure

                    earnings_analysis = {
                        "quarter": (
                            query.split("Q")[1][:1] if "Q" in query else "Unknown"
                        ),
                        "year": "2024",
                        "management_tone": "confident/cautious/concerned",
                        "key_themes": [],
                        "guidance_changes": [],
                        "analyst_sentiment": "positive/neutral/negative",
                    }

                    earnings_analyses.append(earnings_analysis)

                except Exception as e:
                    logger.warning(
                        f"Failed to analyze earnings call for query {query}: {str(e)}"
                    )

            return {
                "symbol": symbol,
                "earnings_call_analyses": earnings_analyses,
                "overall_management_sentiment": "positive",  # Would be calculated
                "guidance_trends": [],
                "key_investor_concerns": [],
            }

        except Exception as e:
            logger.error(f"Error analyzing earnings calls for {symbol}: {str(e)}")
            return {"error": str(e)}

    async def _health_check_impl(self) -> None:
        # Test LocalScrapingService
        health_ok = await self.local_scraping_service.health_check()
        if not health_ok:
            raise Exception("Local scraping service health check failed")

        # Test LLM
        try:
            test_response = await self.llm.ainvoke(
                [
                    HumanMessage(
                        content="Respond with 'OK' if you can process this message."
                    )
                ]
            )
            if "OK" not in test_response.content:
                raise Exception("LLM health check failed")
        except Exception as e:
            raise Exception(f"LLM health check failed: {str(e)}")
