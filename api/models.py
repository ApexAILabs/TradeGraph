from datetime import datetime
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field
from enum import Enum


class APIResponse(BaseModel):
    """Standard API response model."""
    success: bool
    data: Optional[Any] = None
    message: str
    error: Optional[Dict[str, Any]] = None
    timestamp: datetime = Field(default_factory=datetime.now)


class AnalysisRequest(BaseModel):
    """Request model for financial analysis."""
    symbols: List[str] = Field(..., description="List of stock symbols to analyze", example=["AAPL", "MSFT"])
    portfolio_size: Optional[float] = Field(100000, description="Portfolio size in USD", example=100000)
    risk_tolerance: Optional[str] = Field("medium", description="Risk tolerance level", example="medium")
    time_horizon: Optional[str] = Field("medium_term", description="Investment time horizon", example="medium_term")
    include_reports: Optional[bool] = Field(False, description="Include SEC filing analysis", example=False)
    analysis_depth: Optional[str] = Field("standard", description="Analysis depth", example="standard")

    class Config:
        schema_extra = {
            "example": {
                "symbols": ["AAPL", "MSFT", "GOOGL"],
                "portfolio_size": 100000,
                "risk_tolerance": "medium",
                "time_horizon": "medium_term",
                "include_reports": True,
                "analysis_depth": "comprehensive"
            }
        }


class QuickAnalysisRequest(BaseModel):
    """Request model for quick analysis."""
    symbols: List[str] = Field(..., description="List of stock symbols", example=["AAPL", "TSLA"])
    analysis_type: Optional[str] = Field("basic", description="Type of quick analysis", example="basic")


class AlertsRequest(BaseModel):
    """Request model for generating alerts."""
    symbols: List[str] = Field(..., description="List of stock symbols to monitor", example=["AAPL", "MSFT"])
    alert_types: Optional[List[str]] = Field(
        ["price_target", "technical_breakout", "news_event"],
        description="Types of alerts to generate"
    )


class PortfolioOptimizationRequest(BaseModel):
    """Request model for portfolio optimization."""
    symbols: List[str] = Field(..., description="Stock symbols for portfolio")
    constraints: Optional[Dict[str, Any]] = Field(
        default_factory=dict,
        description="Portfolio constraints"
    )
    risk_preferences: Optional[Dict[str, Any]] = Field(
        default_factory=dict,
        description="Risk preferences"
    )
    target_allocation: Optional[Dict[str, float]] = Field(
        default_factory=dict,
        description="Target allocation percentages"
    )


class BackgroundTaskRequest(BaseModel):
    """Request model for background tasks."""
    task_type: str = Field(..., description="Type of background task")
    parameters: Dict[str, Any] = Field(..., description="Task parameters")
    priority: Optional[str] = Field("normal", description="Task priority")


class WebSocketMessage(BaseModel):
    """WebSocket message model."""
    type: str = Field(..., description="Message type")
    data: Any = Field(..., description="Message data")
    timestamp: datetime = Field(default_factory=datetime.now)
    client_id: Optional[str] = Field(None, description="Client identifier")


class AnalysisStatus(str, Enum):
    """Analysis status enumeration."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class AnalysisResult(BaseModel):
    """Analysis result model."""
    analysis_id: str = Field(..., description="Unique analysis identifier")
    status: AnalysisStatus = Field(..., description="Analysis status")
    symbols: List[str] = Field(..., description="Analyzed symbols")
    created_at: datetime = Field(default_factory=datetime.now)
    completed_at: Optional[datetime] = Field(None, description="Completion timestamp")
    results: Optional[Dict[str, Any]] = Field(None, description="Analysis results")
    error_message: Optional[str] = Field(None, description="Error message if failed")
    progress: Optional[float] = Field(None, description="Progress percentage (0-100)")


class RecommendationResponse(BaseModel):
    """Trading recommendation response model."""
    symbol: str
    company_name: str
    recommendation: str  # buy, sell, hold, strong_buy, strong_sell
    confidence_score: float = Field(..., ge=0.0, le=1.0)
    target_price: Optional[float] = None
    stop_loss: Optional[float] = None
    current_price: float
    risk_level: str
    time_horizon: str
    recommended_allocation: float = Field(..., ge=0.0, le=1.0)
    key_factors: List[str] = Field(default_factory=list)
    risks: List[str] = Field(default_factory=list)
    catalysts: List[str] = Field(default_factory=list)
    analyst_notes: Optional[str] = None


class PortfolioResponse(BaseModel):
    """Portfolio recommendation response model."""
    recommendations: List[RecommendationResponse]
    total_confidence: float = Field(..., ge=0.0, le=1.0)
    diversification_score: float = Field(..., ge=0.0, le=1.0)
    overall_risk_level: str
    portfolio_size: float
    expected_return: Optional[float] = None
    expected_volatility: Optional[float] = None
    sector_weights: Dict[str, float] = Field(default_factory=dict)
    rebalancing_frequency: Optional[str] = None


class AlertResponse(BaseModel):
    """Alert response model."""
    symbol: str
    alert_type: str
    message: str
    urgency: str  # low, medium, high, critical
    trigger_conditions: Dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.now)
    expires_at: Optional[datetime] = None


class HealthCheckResponse(BaseModel):
    """Health check response model."""
    status: str
    timestamp: datetime = Field(default_factory=datetime.now)
    version: str
    uptime: Optional[float] = None
    services: Dict[str, str] = Field(default_factory=dict)
    system_info: Optional[Dict[str, Any]] = None


class NewsArticleResponse(BaseModel):
    """News article response model."""
    title: str
    url: str
    content: str
    source: str
    published_at: datetime
    symbols: List[str] = Field(default_factory=list)
    sentiment: Optional[str] = None
    impact_score: Optional[float] = Field(None, ge=0.0, le=1.0)


class MarketDataResponse(BaseModel):
    """Market data response model."""
    symbol: str
    current_price: float
    change: float
    change_percent: float
    volume: int
    market_cap: Optional[float] = None
    pe_ratio: Optional[float] = None
    timestamp: datetime
    provider: Optional[str] = None
    feed_latency_ms: Optional[float] = None


class TechnicalIndicatorsResponse(BaseModel):
    """Technical indicators response model."""
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


class UserPreferences(BaseModel):
    """User preferences model."""
    default_portfolio_size: Optional[float] = 100000
    default_risk_tolerance: Optional[str] = "medium"
    default_time_horizon: Optional[str] = "medium_term"
    preferred_sectors: Optional[List[str]] = Field(default_factory=list)
    excluded_symbols: Optional[List[str]] = Field(default_factory=list)
    notification_preferences: Optional[Dict[str, bool]] = Field(default_factory=dict)
    analysis_preferences: Optional[Dict[str, Any]] = Field(default_factory=dict)


class UserSession(BaseModel):
    """User session model."""
    session_id: str
    user_id: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.now)
    last_activity: datetime = Field(default_factory=datetime.now)
    preferences: Optional[UserPreferences] = None
    analysis_history: List[str] = Field(default_factory=list)  # Analysis IDs


class ErrorResponse(BaseModel):
    """Error response model."""
    error_code: str
    error_message: str
    details: Optional[Dict[str, Any]] = None
    timestamp: datetime = Field(default_factory=datetime.now)
    request_id: Optional[str] = None


class PaginationParams(BaseModel):
    """Pagination parameters."""
    page: int = Field(1, ge=1, description="Page number")
    page_size: int = Field(20, ge=1, le=100, description="Items per page")


class SortParams(BaseModel):
    """Sort parameters."""
    sort_by: Optional[str] = Field(None, description="Field to sort by")
    sort_order: Optional[str] = Field("asc", description="Sort order: asc or desc")


class FilterParams(BaseModel):
    """Filter parameters."""
    filters: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Filter criteria")
    date_range: Optional[Dict[str, datetime]] = Field(None, description="Date range filter")


class AnalysisHistoryResponse(BaseModel):
    """Analysis history response model."""
    analyses: List[AnalysisResult]
    total_count: int
    page: int
    page_size: int
    total_pages: int


class SystemMetrics(BaseModel):
    """System metrics model."""
    cpu_usage: Optional[float] = None
    memory_usage: Optional[float] = None
    active_connections: Optional[int] = None
    total_requests: Optional[int] = None
    average_response_time: Optional[float] = None
    error_rate: Optional[float] = None
    uptime_seconds: Optional[float] = None


class APIMetrics(BaseModel):
    """API metrics model."""
    total_analyses: int = 0
    total_recommendations: int = 0
    total_alerts: int = 0
    active_websocket_connections: int = 0
    system_metrics: Optional[SystemMetrics] = None
    last_updated: datetime = Field(default_factory=datetime.now)
