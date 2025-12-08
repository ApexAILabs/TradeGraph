from datetime import datetime
from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field
from enum import Enum


class RecommendationType(str, Enum):
    BUY = "buy"
    SELL = "sell"
    HOLD = "hold"
    STRONG_BUY = "strong_buy"
    STRONG_SELL = "strong_sell"


class RiskLevel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    VERY_HIGH = "very_high"


class TimeHorizon(str, Enum):
    SHORT_TERM = "short_term"  # 1-3 months
    MEDIUM_TERM = "medium_term"  # 3-12 months
    LONG_TERM = "long_term"  # 1+ years


class TradingRecommendation(BaseModel):
    symbol: str
    company_name: str
    recommendation: RecommendationType
    confidence_score: float = Field(..., ge=0.0, le=1.0)
    target_price: Optional[float] = None
    stop_loss: Optional[float] = None
    current_price: float

    risk_level: RiskLevel
    time_horizon: TimeHorizon

    # Position sizing
    recommended_allocation: float = Field(
        ..., ge=0.0, le=1.0
    )  # Percentage of portfolio
    max_position_size: Optional[float] = None  # Dollar amount

    # Analysis breakdown
    fundamental_score: float = Field(..., ge=0.0, le=1.0)
    technical_score: float = Field(..., ge=0.0, le=1.0)
    sentiment_score: float = Field(..., ge=0.0, le=1.0)

    # Supporting evidence
    key_factors: List[str] = Field(default_factory=list)
    risks: List[str] = Field(default_factory=list)
    catalysts: List[str] = Field(default_factory=list)

    # Metadata
    analysis_timestamp: datetime = Field(default_factory=datetime.now)
    analyst_notes: Optional[str] = None
    sector: Optional[str] = None
    market_cap_category: Optional[str] = None  # "large_cap", "mid_cap", "small_cap"

    # Performance tracking
    expected_return: Optional[float] = None  # Expected return percentage
    expected_volatility: Optional[float] = None
    sharpe_ratio: Optional[float] = None


class PortfolioRecommendation(BaseModel):
    recommendations: List[TradingRecommendation]
    total_confidence: float = Field(..., ge=0.0, le=1.0)
    diversification_score: float = Field(..., ge=0.0, le=1.0)

    # Portfolio metrics
    expected_return: Optional[float] = None
    expected_volatility: Optional[float] = None
    max_drawdown: Optional[float] = None

    # Sector allocation
    sector_weights: Dict[str, float] = Field(default_factory=dict)

    # Risk assessment
    overall_risk_level: RiskLevel
    correlation_matrix: Dict[str, Dict[str, float]] = Field(default_factory=dict)

    # Rebalancing suggestions
    rebalancing_frequency: Optional[str] = None
    next_review_date: Optional[datetime] = None

    generation_timestamp: datetime = Field(default_factory=datetime.now)
    portfolio_size: float = Field(default=100000.0)


class AlertRecommendation(BaseModel):
    symbol: str
    alert_type: str  # "price_target", "news_event", "technical_breakout"
    message: str
    urgency: str  # "low", "medium", "high", "critical"
    trigger_conditions: Dict[str, Any] = Field(default_factory=dict)
    expiry_date: Optional[datetime] = None
    created_at: datetime = Field(default_factory=datetime.now)
