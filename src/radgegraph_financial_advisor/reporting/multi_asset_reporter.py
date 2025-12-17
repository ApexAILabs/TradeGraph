"""PDF builder for multi-asset allocation plans."""

from __future__ import annotations

import os
from datetime import datetime
from typing import Any, Dict, List, Optional
import textwrap

from reportlab.lib.pagesizes import LETTER
from reportlab.lib.units import inch
from reportlab.pdfgen import canvas


class MultiAssetPDFReportWriter:
    """Renders allocation plans across strategies/horizons into a PDF."""

    def __init__(self) -> None:
        self.page_width, self.page_height = LETTER
        self.margin = 0.75 * inch
        self.line_height = 14

    def build_report(
        self,
        *,
        plan: Dict[str, Any],
        output_path: Optional[str] = None,
    ) -> str:
        os.makedirs("results", exist_ok=True)
        if not output_path:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = os.path.join(
                "results", f"radgegraph_multi_asset_{timestamp}.pdf"
            )

        doc = canvas.Canvas(output_path, pagesize=LETTER)
        cursor_y = self.page_height - self.margin

        cursor_y = self._draw_title(doc, cursor_y, "Multi-Asset Allocation Blueprint")
        cursor_y = self._draw_subtitle(
            doc,
            cursor_y,
            f"Budget: ${plan.get('budget', 0):,.2f} | Generated {datetime.now():%Y-%m-%d %H:%M UTC}",
        )

        notes = plan.get("notes") or []
        if notes:
            cursor_y = self._draw_bullet_section(doc, cursor_y, "Advisor Notes", notes)

        for strategy in plan.get("strategies", []):
            cursor_y = self._draw_strategy_section(doc, cursor_y, strategy)

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

    def _draw_strategy_section(
        self,
        doc: canvas.Canvas,
        cursor_y: float,
        strategy: Dict[str, Any],
    ) -> float:
        cursor_y = self._ensure_space(doc, cursor_y, min_height=140)
        title = (
            f"Strategy: {str(strategy.get('strategy', '')).title()} - "
            f"{strategy.get('description', '')}"
        )
        doc.setFont("Helvetica-Bold", 14)
        doc.drawString(self.margin, cursor_y, title)
        cursor_y -= 18

        horizons = strategy.get("horizons", {})
        for horizon_key, payload in horizons.items():
            cursor_y = self._draw_horizon(doc, cursor_y, horizon_key, payload)
        return cursor_y

    def _draw_horizon(
        self,
        doc: canvas.Canvas,
        cursor_y: float,
        horizon_key: str,
        payload: Dict[str, Any],
    ) -> float:
        cursor_y = self._ensure_space(doc, cursor_y, min_height=80)
        label = payload.get("label", horizon_key)
        doc.setFont("Helvetica-Bold", 12)
        doc.drawString(
            self.margin,
            cursor_y,
            f"{label}: {payload.get('risk_focus', 'Risk focus n/a')}",
        )
        cursor_y -= 14
        doc.setFont("Helvetica", 10)
        for allocation in payload.get("allocations", []):
            amount = allocation.get("amount", 0)
            weight = allocation.get("weight", 0) * 100
            doc.drawString(
                self.margin + 10,
                cursor_y,
                f"- {allocation.get('asset_class', '').upper()}: {weight:.1f}% (${amount:,.2f})",
            )
            cursor_y -= self.line_height
            rationale = allocation.get("rationale")
            if rationale:
                for line in self._wrap_text(f"Rationale: {rationale}", 92):
                    doc.drawString(self.margin + 20, cursor_y, line)
                    cursor_y -= self.line_height
            samples = allocation.get("sample_assets", [])
            if samples:
                sample_line = ", ".join(
                    f"{item.get('symbol')} ({item.get('thesis')})" for item in samples
                )
                for line in self._wrap_text(f"Sample: {sample_line}", 92):
                    doc.drawString(self.margin + 20, cursor_y, line)
                    cursor_y -= self.line_height
            cursor_y -= 4
        return cursor_y

    def _draw_bullet_section(
        self,
        doc: canvas.Canvas,
        cursor_y: float,
        title: str,
        bullets: List[str],
    ) -> float:
        cursor_y = self._ensure_space(doc, cursor_y, min_height=70)
        doc.setFont("Helvetica-Bold", 14)
        doc.drawString(self.margin, cursor_y, title)
        cursor_y -= 18
        doc.setFont("Helvetica", 11)
        for bullet in bullets:
            for line in self._wrap_text(bullet, 94):
                doc.drawString(self.margin + 10, cursor_y, f"• {line}")
                cursor_y -= self.line_height
        return cursor_y - 6

    def _ensure_space(
        self, doc: canvas.Canvas, cursor_y: float, *, min_height: float
    ) -> float:
        if cursor_y - min_height <= self.margin:
            doc.showPage()
            cursor_y = self.page_height - self.margin
        return cursor_y

    def _wrap_text(self, text: str, width: int) -> List[str]:
        if not text:
            return [""]
        return textwrap.wrap(text, width=width) or [text]


__all__ = ["MultiAssetPDFReportWriter"]
