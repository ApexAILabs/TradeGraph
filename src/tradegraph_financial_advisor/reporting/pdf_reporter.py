"""PDF utilities to render multi-channel financial reports."""

from __future__ import annotations

import os
from datetime import datetime
from typing import Any, Dict, List, Optional
import textwrap

from reportlab.lib.pagesizes import LETTER
from reportlab.lib.units import inch
from reportlab.lib.utils import ImageReader
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
        portfolio_recommendation: Optional[Dict[str, Any]] = None,
        analysis_summary: Optional[Dict[str, Any]] = None,
        allocation_chart_path: Optional[str] = None,
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

        cursor_y = self._draw_portfolio_overview(
            doc,
            cursor_y,
            analysis_summary or {},
            portfolio_recommendation or {},
            allocation_chart_path,
        )

        cursor_y = self._draw_section(
            doc, cursor_y, "Executive Summary", summary_payload.get("summary_text", "")
        )

        cursor_y = self._draw_section(
            doc, cursor_y, "Desk Memo", summary_payload.get("advisor_memo", "")
        )

        cursor_y = self._draw_bullet_section(
            doc,
            cursor_y,
            "News Highlights",
            summary_payload.get("news_takeaways", []),
        )

        cursor_y = self._draw_bullet_section(
            doc,
            cursor_y,
            "Price & Trend Signals",
            summary_payload.get("price_action_notes", []),
        )

        cursor_y = self._draw_bullet_section(
            doc,
            cursor_y,
            "Actionable Guidance",
            summary_payload.get("guidance_points", []),
        )

        cursor_y = self._draw_section(
            doc,
            cursor_y,
            "Risk & Signals",
            f"Suggested Stance: {summary_payload.get('buy_or_sell_view', 'n/a').upper()}\n"
            f"Risk Assessment: {summary_payload.get('risk_assessment', 'n/a')}\n"
            f"Trend Notes: {summary_payload.get('trend_commentary', 'n/a')}",
        )

        cursor_y = self._draw_key_stats(
            doc, cursor_y, summary_payload.get("key_stats", {})
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

    def _draw_bullet_section(
        self,
        doc: canvas.Canvas,
        cursor_y: float,
        title: str,
        items: List[str],
    ) -> float:
        if not items:
            return cursor_y
        cursor_y = self._ensure_space(doc, cursor_y, min_height=80)
        doc.setFont("Helvetica-Bold", 14)
        doc.drawString(self.margin, cursor_y, title)
        cursor_y -= 18
        doc.setFont("Helvetica", 11)
        for item in items:
            for line in self._wrap_text(item, 92):
                doc.drawString(self.margin + 10, cursor_y, f"• {line}")
                cursor_y -= self.line_height
        return cursor_y - 6

    def _draw_key_stats(
        self,
        doc: canvas.Canvas,
        cursor_y: float,
        stats: Dict[str, Any],
    ) -> float:
        if not stats:
            return cursor_y
        cursor_y = self._ensure_space(doc, cursor_y, min_height=70)
        doc.setFont("Helvetica-Bold", 14)
        doc.drawString(self.margin, cursor_y, "Monitoring Stats")
        cursor_y -= 18
        doc.setFont("Helvetica", 11)
        lines = [
            f"Channels monitored: {stats.get('channel_count', 0)}",
            f"Headlines ingested: {stats.get('headline_count', 0)}",
            f"Recommendations referenced: {stats.get('recommendation_count', 0)}",
        ]
        covered = stats.get("covered_symbols")
        if covered:
            lines.append(f"Symbols highlighted: {', '.join(covered)}")
        for line in lines:
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
            highlights = [
                item.get("title", "") for item in payload.get("items", [])[:3]
            ]
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

    def _draw_portfolio_overview(
        self,
        doc: canvas.Canvas,
        cursor_y: float,
        analysis_summary: Dict[str, Any],
        portfolio_recommendation: Dict[str, Any],
        allocation_chart_path: Optional[str],
    ) -> float:
        cursor_y = self._ensure_space(doc, cursor_y, min_height=180)
        section_top = cursor_y
        doc.setFont("Helvetica-Bold", 14)
        doc.drawString(self.margin, cursor_y, "Portfolio Overview")
        cursor_y -= 18
        doc.setFont("Helvetica", 11)

        portfolio_size = analysis_summary.get("portfolio_size")
        risk_tolerance = analysis_summary.get("risk_tolerance", "-")
        time_horizon = analysis_summary.get("time_horizon", "-")
        symbols_line = ", ".join(analysis_summary.get("symbols_analyzed", []))
        text_lines = [
            f"Portfolio Size: {self._format_currency(portfolio_size)}",
            f"Risk Tolerance: {risk_tolerance.title() if isinstance(risk_tolerance, str) else risk_tolerance}",
            f"Time Horizon: {time_horizon.replace('_', ' ').title() if isinstance(time_horizon, str) else time_horizon}",
        ]
        if symbols_line:
            text_lines.append(f"Focus Symbols: {symbols_line}")

        total_conf = portfolio_recommendation.get("total_confidence")
        diversification = portfolio_recommendation.get("diversification_score")
        expected_return = portfolio_recommendation.get("expected_return")
        expected_vol = portfolio_recommendation.get("expected_volatility")
        overall_risk = portfolio_recommendation.get("overall_risk_level")
        if isinstance(total_conf, (int, float)):
            text_lines.append(f"Portfolio Confidence: {total_conf:.0%}")
        if isinstance(diversification, (int, float)):
            text_lines.append(f"Diversification Score: {diversification:.0%}")
        if isinstance(expected_return, (int, float)):
            text_lines.append(f"Expected Return: {expected_return:.1%}")
        elif expected_return:
            text_lines.append(f"Expected Return: {expected_return}")
        if isinstance(expected_vol, (int, float)):
            text_lines.append(f"Expected Volatility: {expected_vol:.1%}")
        elif expected_vol:
            text_lines.append(f"Expected Volatility: {expected_vol}")
        if overall_risk:
            text_lines.append(f"Overall Risk: {str(overall_risk).title()}")

        for line in text_lines:
            doc.drawString(self.margin, cursor_y, line)
            cursor_y -= self.line_height

        cursor_after_text = cursor_y - 6

        chart_bottom = cursor_after_text
        if allocation_chart_path and os.path.exists(allocation_chart_path):
            try:
                image = ImageReader(allocation_chart_path)
                chart_width = 2.8 * inch
                chart_height = 2.8 * inch
                chart_x = self.page_width - self.margin - chart_width
                chart_y = section_top - chart_height
                doc.drawImage(
                    image,
                    chart_x,
                    chart_y,
                    width=chart_width,
                    height=chart_height,
                    preserveAspectRatio=True,
                    mask="auto",
                )
                chart_bottom = min(chart_y - 10, cursor_after_text)
            except Exception:
                chart_bottom = cursor_after_text

        return min(cursor_after_text, chart_bottom)

    def _ensure_space(
        self, doc: canvas.Canvas, cursor_y: float, *, min_height: float
    ) -> float:
        if cursor_y - min_height <= self.margin:
            doc.showPage()
            cursor_y = self.page_height - self.margin
        return cursor_y

    def _wrap_text(self, text: Any, width: int) -> List[str]:
        """Safely wrap arbitrary content for drawing in the PDF."""
        if text is None:
            return ["n/a"]

        normalized: str
        if isinstance(text, str):
            normalized = text.strip()
        elif isinstance(text, (list, tuple)):
            flattened = []
            for item in text:
                if item is None:
                    continue
                if isinstance(item, str):
                    flattened.append(item.strip())
                elif isinstance(item, dict):
                    flattened.append(
                        ", ".join(f"{k}: {v}" for k, v in item.items() if v is not None)
                    )
                else:
                    flattened.append(str(item))
            normalized = "; ".join(filter(None, flattened))
        elif isinstance(text, dict):
            normalized = ", ".join(
                f"{k}: {v}" for k, v in text.items() if v is not None
            )
        else:
            normalized = str(text)

        normalized = normalized.strip()
        if not normalized:
            return ["n/a"]

        wrapped = textwrap.wrap(normalized, width=width)
        return wrapped or [normalized]

    @staticmethod
    def _format_pct(trend: Optional[Dict[str, Any]]) -> str:
        if not trend:
            return " - "
        percent = trend.get("percent_change")
        if percent is None:
            return " - "
        return f"{percent:+.1f}%"

    @staticmethod
    def _format_currency(value: Optional[float]) -> str:
        if value is None:
            return "n/a"
        try:
            return f"${float(value):,.0f}"
        except (TypeError, ValueError):
            return str(value)


__all__ = ["ChannelPDFReportWriter"]
