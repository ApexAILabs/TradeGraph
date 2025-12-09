"""Agent that summarizes streaming channel data for PDF reports."""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from loguru import logger
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI

from .base_agent import BaseAgent
from ..config.settings import settings
from ..utils.helpers import generate_summary


class ChannelReportAgent(BaseAgent):
    """Synthesizes channel payloads and market trends into a narrative."""

    def __init__(
        self,
        *,
        llm_model_name: str = "gpt-5-nano",
        llm_client: Optional[ChatOpenAI] = None,
        enable_llm: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            name="ChannelReportAgent",
            description="Summarizes websocket channel data for investor-ready narratives",
            **kwargs,
        )
        self.llm = llm_client
        if not self.llm and enable_llm and settings.openai_api_key:
            self.llm = ChatOpenAI(
                model=llm_model_name,
                temperature=0.1,
                api_key=settings.openai_api_key,
            )

    async def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        channel_payloads = input_data.get("channel_payloads", {})
        price_trends = input_data.get("price_trends", {})
        recommendations = input_data.get("recommendations", [])

        fallback = self._build_fallback_summary(
            channel_payloads, price_trends, recommendations
        )

        if not self.llm:
            return fallback

        try:
            prompt_payload = {
                "channel_payloads": channel_payloads,
                "price_trends": price_trends,
                "recommendations": recommendations,
            }
            prompt = (
                "You are TradeGraph's senior analyst. Combine the multichannel news feeds, "
                "risk context, and trend data below into a concise JSON summary with "
                "keys: news_takeaways (list of strings), risk_assessment (string), "
                "buy_or_sell_view (string), trend_commentary (string), key_stats (object), "
                "and summary_text (string)."
            )
            response = await self.llm.ainvoke(
                [HumanMessage(content=f"{prompt}\nINPUT:\n{json.dumps(prompt_payload)[:6000]}")]
            )
            data = json.loads(response.content)
            fallback.update({k: v for k, v in data.items() if v})
            return fallback
        except Exception as exc:  # pragma: no cover - network/LLM variability
            logger.warning(f"ChannelReportAgent LLM summary failed: {exc}")
            return fallback

    def _build_fallback_summary(
        self,
        channel_payloads: Dict[str, Any],
        price_trends: Dict[str, Any],
        recommendations: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        news_takeaways: List[str] = []
        for channel_id, payload in channel_payloads.items():
            items = payload.get("items", [])
            if not items:
                continue
            titles = ", ".join(item.get("title", "").strip() for item in items[:2])
            news_takeaways.append(
                f"{payload.get('title', channel_id)} highlights: {titles}"
            )

        risk_counts: Dict[str, int] = {}
        for rec in recommendations:
            risk = rec.get("risk_level", "unknown")
            risk_counts[risk] = risk_counts.get(risk, 0) + 1

        buy_view = "hold"
        buy_votes = sum(1 for rec in recommendations if "buy" in str(rec.get("recommendation", "")).lower())
        sell_votes = sum(1 for rec in recommendations if "sell" in str(rec.get("recommendation", "")).lower())
        if buy_votes > sell_votes:
            buy_view = "buy"
        elif sell_votes > buy_votes:
            buy_view = "reduce"

        trend_commentary = self._summarize_trends(price_trends)
        summary_text = generate_summary(" ".join(news_takeaways)) or (
            "Latest headlines aggregated across equity, crypto, and free news agencies."
        )

        return {
            "news_takeaways": news_takeaways,
            "risk_assessment": f"Risk mix: {risk_counts or {'unknown': 0}}",
            "buy_or_sell_view": buy_view,
            "trend_commentary": trend_commentary,
            "key_stats": {
                "recommendation_count": len(recommendations),
                "channels": list(channel_payloads.keys()),
            },
            "summary_text": summary_text,
        }

    def _summarize_trends(self, price_trends: Dict[str, Any]) -> str:
        if not price_trends:
            return "Trend data unavailable."
        phrases = []
        for symbol, payload in price_trends.items():
            trends = payload.get("trends", {})
            month = trends.get("last_month", {}).get("percent_change")
            week = trends.get("last_week", {}).get("percent_change")
            day = trends.get("last_day", {}).get("percent_change")
            hour = trends.get("last_hour", {}).get("percent_change")
            parts = []
            if month is not None:
                parts.append(f"{month:+.1f}% 1M")
            if week is not None:
                parts.append(f"{week:+.1f}% 1W")
            if day is not None:
                parts.append(f"{day:+.1f}% 1D")
            if hour is not None:
                parts.append(f"{hour:+.1f}% 1H")
            if parts:
                phrases.append(f"{symbol}: {' / '.join(parts)} (month-to-date view)")
        return "; ".join(phrases) if phrases else "Trend data unavailable."


__all__ = ["ChannelReportAgent"]
