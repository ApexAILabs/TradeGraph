"""PDF utilities to render multi-channel financial reports."""

from __future__ import annotations

import os
from datetime import datetime
from typing import Any, Dict, List, Optional
import textwrap

from reportlab.lib.pagesizes import LETTER
from reportlab.lib.units import inch
from reportlab.pdfgen import canvas


class ChannelPDFReportWriter:
    """Minimal PDF builder for the multichannel financial report."""

    def __init__(self) -> None:
        self.page_width, self.page_height = LETTER
        self.margin = 0.75 * inch
        self.line_height = 14

    def build_report(
        self,
        *,
        summary_payload: Dict[str, Any],
        channel_payloads: Dict[str, Any],
        price_trends: Dict[str, Any],
        recommendations: List[Dict[str, Any]],
        symbols: List[str],
        output_path: Optional[str] = None,
    ) -> str:
        os.makedirs("results", exist_ok=True)
        if not output_path:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = os.path.join(
                "results", f"tradegraph_multichannel_{timestamp}.pdf"
            )

        doc = canvas.Canvas(output_path, pagesize=LETTER)
        cursor_y = self.page_height - self.margin

        cursor_y = self._draw_title(doc, cursor_y, "TradeGraph Multichannel Report")
        cursor_y = self._draw_subtitle(
            doc,
            cursor_y,
            f"Symbols: {', '.join(symbols)} | Generated {datetime.now():%Y-%m-%d %H:%M UTC}",
        )

        cursor_y = self._draw_section(
            doc, cursor_y, "Executive Summary", summary_payload.get("summary_text", "")
        )

        news_text = "\n".join(summary_payload.get("news_takeaways", []))
        cursor_y = self._draw_section(doc, cursor_y, "News Highlights", news_text)

        cursor_y = self._draw_section(
            doc,
            cursor_y,
            "Risk & Signals",
            f"Suggested Stance: {summary_payload.get('buy_or_sell_view', 'n/a').upper()}\n"
            f"Risk Assessment: {summary_payload.get('risk_assessment', 'n/a')}\n"
            f"Trend Notes: {summary_payload.get('trend_commentary', 'n/a')}",
        )

        cursor_y = self._draw_channel_breakdown(doc, cursor_y, channel_payloads)
        cursor_y = self._draw_recommendations(doc, cursor_y, recommendations)
        cursor_y = self._draw_trends(doc, cursor_y, price_trends)

        doc.save()
        return output_path

    def _draw_title(self, doc: canvas.Canvas, cursor_y: float, text: str) -> float:
        doc.setFont("Helvetica-Bold", 20)
        doc.drawString(self.margin, cursor_y, text)
        return cursor_y - 24

    def _draw_subtitle(self, doc: canvas.Canvas, cursor_y: float, text: str) -> float:
        doc.setFont("Helvetica", 11)
        doc.drawString(self.margin, cursor_y, text)
        return cursor_y - 18

    def _draw_section(
        self, doc: canvas.Canvas, cursor_y: float, title: str, body: str
    ) -> float:
        cursor_y = self._ensure_space(doc, cursor_y, min_height=80)
        doc.setFont("Helvetica-Bold", 14)
        doc.drawString(self.margin, cursor_y, title)
        cursor_y -= 18
        doc.setFont("Helvetica", 11)
        wrapped = self._wrap_text(body, 96)
        for line in wrapped:
            doc.drawString(self.margin, cursor_y, line)
            cursor_y -= self.line_height
        return cursor_y - 6

    def _draw_channel_breakdown(
        self, doc: canvas.Canvas, cursor_y: float, channels: Dict[str, Any]
    ) -> float:
        cursor_y = self._ensure_space(doc, cursor_y, min_height=120)
        doc.setFont("Helvetica-Bold", 14)
        doc.drawString(self.margin, cursor_y, "Channel Breakdown")
        cursor_y -= 18
        doc.setFont("Helvetica", 10)

        for channel_id, payload in channels.items():
            cursor_y = self._ensure_space(doc, cursor_y, min_height=60)
            doc.setFont("Helvetica-Bold", 12)
            doc.drawString(self.margin, cursor_y, payload.get("title", channel_id))
            cursor_y -= 14
            doc.setFont("Helvetica", 10)
            highlights = [item.get("title", "") for item in payload.get("items", [])[:3]]
            body = "; ".join(highlights) or "No items collected"
            for line in self._wrap_text(body, 90):
                doc.drawString(self.margin + 10, cursor_y, f"- {line}")
                cursor_y -= self.line_height
        return cursor_y

    def _draw_recommendations(
        self, doc: canvas.Canvas, cursor_y: float, recommendations: List[Dict[str, Any]]
    ) -> float:
        if not recommendations:
            return cursor_y
        cursor_y = self._ensure_space(doc, cursor_y, min_height=80)
        doc.setFont("Helvetica-Bold", 14)
        doc.drawString(self.margin, cursor_y, "Trading Recommendations")
        cursor_y -= 18
        doc.setFont("Helvetica", 10)
        for rec in recommendations:
            cursor_y = self._ensure_space(doc, cursor_y, min_height=50)
            symbol_line = (
                f"{rec.get('symbol')} | {rec.get('recommendation', '').upper()} | "
                f"Risk: {rec.get('risk_level', 'n/a')} | Confidence: {rec.get('confidence_score', 0):.2f}"
            )
            doc.drawString(self.margin, cursor_y, symbol_line)
            cursor_y -= self.line_height
            details = (
                f"Target ${rec.get('target_price', 'n/a')} | Stop ${rec.get('stop_loss', 'n/a')} | "
                f"Allocation {rec.get('recommended_allocation', 0):.1%}"
            )
            doc.drawString(self.margin, cursor_y, details)
            cursor_y -= self.line_height
            factors = ", ".join(rec.get("key_factors", [])[:2])
            if factors:
                doc.drawString(self.margin, cursor_y, f"Factors: {factors}")
                cursor_y -= self.line_height
            cursor_y -= 4
        return cursor_y

    def _draw_trends(
        self,
        doc: canvas.Canvas,
        cursor_y: float,
        price_trends: Dict[str, Any],
    ) -> float:
        if not price_trends:
            return cursor_y
        cursor_y = self._ensure_space(doc, cursor_y, min_height=120)
        doc.setFont("Helvetica-Bold", 14)
        doc.drawString(self.margin, cursor_y, "Trend Snapshots")
        cursor_y -= 18
        doc.setFont("Helvetica", 9)
        doc.drawString(
            self.margin,
            cursor_y,
            "Month-to-date focus: showing 1M, 1W, 1D, and 1H moves from the latest pricing.",
        )
        cursor_y -= self.line_height
        doc.setFont("Helvetica", 10)

        headers = "Symbol      1M %     1W %     1D %     1H %"
        doc.drawString(self.margin, cursor_y, headers)
        cursor_y -= self.line_height
        for symbol, payload in price_trends.items():
            trends = payload.get("trends", {})
            row = (
                f"{symbol:<10}"
                f"{self._format_pct(trends.get('last_month')):<9}"
                f"{self._format_pct(trends.get('last_week')):<9}"
                f"{self._format_pct(trends.get('last_day')):<9}"
                f"{self._format_pct(trends.get('last_hour')):<9}"
            )
            doc.drawString(self.margin, cursor_y, row)
            cursor_y -= self.line_height
        return cursor_y

    def _ensure_space(
        self, doc: canvas.Canvas, cursor_y: float, *, min_height: float
    ) -> float:
        if cursor_y - min_height <= self.margin:
            doc.showPage()
            cursor_y = self.page_height - self.margin
        return cursor_y

    def _wrap_text(self, text: str, width: int) -> List[str]:
        if not text:
            return ["n/a"]
        return textwrap.wrap(text, width=width) or [text]

    @staticmethod
    def _format_pct(trend: Optional[Dict[str, Any]]) -> str:
        if not trend:
            return " - "
        percent = trend.get("percent_change")
        if percent is None:
            return " - "
        return f"{percent:+.1f}%"


__all__ = ["ChannelPDFReportWriter"]
