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
        channel_payloads = self._filter_channel_payloads(
            input_data.get("channel_payloads", {})
        )
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
                "recommendation_snapshot": self._summarize_recommendations(
                    recommendations
                ),
            }
            prompt = (
                "You are TradeGraph's senior portfolio strategist. Blend the curated news "
                "feeds, short-term price action, and active recommendations into guidance "
                "for an informed investor. Respond in JSON with keys: summary_text (2 "
                "advisor-style paragraphs), advisor_memo (actionable paragraph), "
                "news_takeaways (list of 3 strings), guidance_points (list of next "
                "actions), risk_assessment, buy_or_sell_view, trend_commentary, "
                "price_action_notes (list), and key_stats (object describing counts)."
            )
            response = await self.llm.ainvoke(
                [
                    HumanMessage(
                        content=f"{prompt}\nINPUT:\n{json.dumps(prompt_payload)[:6000]}"
                    )
                ]
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

        key_stats = self._build_key_stats(channel_payloads, recommendations)

        risk_counts: Dict[str, int] = {}
        for rec in recommendations:
            risk = rec.get("risk_level", "unknown")
            risk_counts[risk] = risk_counts.get(risk, 0) + 1

        buy_view = "hold"
        buy_votes = sum(
            1
            for rec in recommendations
            if "buy" in str(rec.get("recommendation", "")).lower()
        )
        sell_votes = sum(
            1
            for rec in recommendations
            if "sell" in str(rec.get("recommendation", "")).lower()
        )
        if buy_votes > sell_votes:
            buy_view = "buy"
        elif sell_votes > buy_votes:
            buy_view = "reduce"

        trend_commentary = self._summarize_trends(price_trends)
        price_action_notes = self._build_price_notes(price_trends)
        guidance_points = self._build_guidance_points(
            recommendations, price_action_notes
        )

        narrative_seed = " ".join(news_takeaways[:3]) or "Mixed market color."
        summary_text = (
            generate_summary(narrative_seed)
            or "Fresh headlines suggest a balanced tape across equities and crypto."
        )
        advisor_memo = (
            f"My read: {buy_view.upper()} bias while monitoring {risk_counts or {'unknown': 0}}. "
            f"Trend check: {trend_commentary}."
        )

        return {
            "news_takeaways": news_takeaways,
            "risk_assessment": f"Risk mix: {risk_counts or {'unknown': 0}}",
            "buy_or_sell_view": buy_view,
            "trend_commentary": trend_commentary,
            "key_stats": key_stats,
            "summary_text": summary_text,
            "advisor_memo": advisor_memo,
            "price_action_notes": price_action_notes,
            "guidance_points": guidance_points,
        }

    def _filter_channel_payloads(
        self, channel_payloads: Dict[str, Any]
    ) -> Dict[str, Any]:
        return {
            channel_id: payload
            for channel_id, payload in channel_payloads.items()
            if channel_id != "open_source_agencies"
        }

    def _build_key_stats(
        self,
        channel_payloads: Dict[str, Any],
        recommendations: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        total_items = sum(
            len(payload.get("items", [])) for payload in channel_payloads.values()
        )
        covered_symbols = {
            symbol
            for payload in channel_payloads.values()
            for item in payload.get("items", [])
            for symbol in item.get("matched_symbols", [])
            if symbol
        }
        return {
            "channel_count": len(channel_payloads),
            "headline_count": total_items,
            "recommendation_count": len(recommendations),
            "covered_symbols": sorted(covered_symbols),
        }

    def _build_price_notes(self, price_trends: Dict[str, Any]) -> List[str]:
        notes: List[str] = []
        for symbol, payload in price_trends.items():
            trends = payload.get("trends", {})
            day = trends.get("last_day", {}).get("percent_change")
            week = trends.get("last_week", {}).get("percent_change")
            hour = trends.get("last_hour", {}).get("percent_change")
            pieces = []
            if week is not None:
                pieces.append(f"{week:+.1f}% weekly")
            if day is not None:
                pieces.append(f"{day:+.1f}% daily")
            if hour is not None:
                pieces.append(f"{hour:+.1f}% hourly")
            if pieces:
                notes.append(f"{symbol}: {' / '.join(pieces)} post-close move")
        return notes

    def _build_guidance_points(
        self,
        recommendations: List[Dict[str, Any]],
        price_notes: List[str],
    ) -> List[str]:
        guidance: List[str] = []
        for rec in recommendations[:3]:
            symbol = rec.get("symbol", "")
            rec_text = rec.get("recommendation", "hold").replace("_", " ")
            allocation = rec.get("recommended_allocation")
            allocation_text = (
                f"targeting {allocation:.1%} weight"
                if isinstance(allocation, (int, float))
                else ""
            )
            note = rec.get("analyst_notes") or ", ".join(rec.get("key_factors", [])[:2])
            clause = f"{symbol}: {rec_text.title()} {allocation_text}".strip()
            if note:
                clause = f"{clause} — {note}"
            guidance.append(clause)

        if price_notes:
            guidance.append(f"Monitor price tape: {price_notes[0]}")
        return guidance

    def _summarize_recommendations(
        self, recommendations: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        if not recommendations:
            return {"total": 0}

        counts: Dict[str, int] = {}
        top_symbols: List[str] = []
        highest_conf = sorted(
            recommendations,
            key=lambda rec: rec.get("confidence_score", 0),
            reverse=True,
        )[:3]
        for rec in recommendations:
            name = str(rec.get("recommendation", "unknown")).lower()
            counts[name] = counts.get(name, 0) + 1
            if rec.get("symbol"):
                top_symbols.append(rec["symbol"])

        return {
            "total": len(recommendations),
            "counts": counts,
            "top_conviction": [
                {
                    "symbol": rec.get("symbol"),
                    "confidence_score": rec.get("confidence_score"),
                    "recommendation": rec.get("recommendation"),
                }
                for rec in highest_conf
            ],
            "symbols": top_symbols,
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
