"""
RADGEGRAPH – Risk and Asset Decision Engine Based on AGENTS, AI, and Knowledge Graph

A multi-agent financial analysis system using LangGraph, Firecrawl, and MCP
for intelligent trading recommendations based on real-time financial news
and company performance analysis.
"""

__version__ = "1.0.0"
__author__ = "RADGEGRAPH Team"

from .main import FinancialAdvisor
from .config.settings import Settings
from .models.recommendations import TradingRecommendation, RecommendationType

__all__ = [
    "FinancialAdvisor",
    "Settings",
    "TradingRecommendation",
    "RecommendationType",
]
