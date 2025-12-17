from datetime import datetime
from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field
from enum import Enum


class SentimentType(str, Enum):
    BULLISH = "bullish"
    BEARISH = "bearish"
    NEUTRAL = "neutral"


class NewsArticle(BaseModel):
    title: str
    url: str
    content: str
    summary: Optional[str] = None
    source: str
    published_at: datetime
    symbols: List[str] = Field(default_factory=list)
    sentiment: Optional[SentimentType] = None
    impact_score: Optional[float] = Field(None, ge=0.0, le=1.0)


class CompanyFinancials(BaseModel):
    symbol: str
    company_name: str
    market_cap: Optional[float] = None
    pe_ratio: Optional[float] = None
    eps: Optional[float] = None
    revenue: Optional[float] = None
    net_income: Optional[float] = None
    debt_to_equity: Optional[float] = None
    current_ratio: Optional[float] = None
    return_on_equity: Optional[float] = None
    return_on_assets: Optional[float] = None
    price_to_book: Optional[float] = None
    dividend_yield: Optional[float] = None
    beta: Optional[float] = None
    fifty_two_week_high: Optional[float] = None
    fifty_two_week_low: Optional[float] = None
    current_price: Optional[float] = None
    report_date: Optional[datetime] = None
    report_type: Optional[str] = None  # "quarterly", "annual"


class MarketData(BaseModel):
    symbol: str
    current_price: float
    change: float
    change_percent: float
    volume: int
    market_cap: Optional[float] = None
    pe_ratio: Optional[float] = None
    timestamp: datetime


class TechnicalIndicators(BaseModel):
    symbol: str
    sma_20: Optional[float] = None
    sma_50: Optional[float] = None
    ema_12: Optional[float] = None
    ema_26: Optional[float] = None
    rsi: Optional[float] = None
    macd: Optional[float] = None
    macd_signal: Optional[float] = None
    bollinger_upper: Optional[float] = None
    bollinger_lower: Optional[float] = None
    support_level: Optional[float] = None
    resistance_level: Optional[float] = None
    timestamp: datetime


class AnalysisContext(BaseModel):
    symbol: str
    news_articles: List[NewsArticle] = Field(default_factory=list)
    financials: Optional[CompanyFinancials] = None
    market_data: Optional[MarketData] = None
    technical_indicators: Optional[TechnicalIndicators] = None
    peer_comparison: Dict[str, Any] = Field(default_factory=dict)
    sector_performance: Dict[str, Any] = Field(default_factory=dict)
    analysis_timestamp: datetime = Field(default_factory=datetime.now)
