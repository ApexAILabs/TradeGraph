import asyncio
from typing import Any, Dict, List, Optional, Callable
from datetime import datetime
import json
import os
import re
from loguru import logger


def validate_symbols(symbols: List[str]) -> List[str]:
    """
    Validate and clean stock symbols.

    Args:
        symbols: List of stock symbols

    Returns:
        List of validated symbols
    """
    validated = []

    for symbol in symbols:
        # Clean and validate symbol
        clean_symbol = symbol.strip().upper()

        # Basic validation
        if len(clean_symbol) >= 1 and len(clean_symbol) <= 5 and clean_symbol.isalpha():
            validated.append(clean_symbol)
        else:
            logger.warning(f"Invalid symbol format: {symbol}")

    return validated


def format_currency(amount: float, decimals: int = 2) -> str:
    """Format currency with proper commas and decimals."""
    return f"${amount:,.{decimals}f}"


def format_percentage(value: float, decimals: int = 1) -> str:
    """Format percentage with proper symbol."""
    return f"{value * 100:.{decimals}f}%"


def generate_summary(text: str, max_sentences: int = 2, max_length: int = 320) -> str:
    """Create a concise summary using the leading sentences of the content."""
    if not text:
        return ""

    normalized = re.sub(r"\s+", " ", text).strip()
    if not normalized:
        return ""

    sentences = re.split(r"(?<=[.!?])\s+", normalized)
    summary = " ".join(sentences[:max_sentences]).strip()

    if not summary:
        summary = normalized[:max_length].strip()
        if len(normalized) > max_length:
            summary = summary.rstrip(" ,;:-") + "..."
        return summary

    if len(summary) > max_length:
        truncated = summary[:max_length].rstrip()
        last_space = truncated.rfind(" ")
        if last_space > 0:
            truncated = truncated[:last_space]
        summary = truncated.rstrip(" ,;:-") + "..."

    return summary


def calculate_portfolio_metrics(
    recommendations: List[Dict[str, Any]],
) -> Dict[str, float]:
    """
    Calculate basic portfolio metrics from recommendations.

    Args:
        recommendations: List of recommendation dictionaries

    Returns:
        Dictionary of portfolio metrics
    """
    if not recommendations:
        return {}

    total_allocation = sum(
        rec.get("recommended_allocation", 0) for rec in recommendations
    )
    avg_confidence = sum(
        rec.get("confidence_score", 0) for rec in recommendations
    ) / len(recommendations)

    # Risk distribution
    risk_counts = {}
    for rec in recommendations:
        risk = rec.get("risk_level", "medium")
        risk_counts[risk] = risk_counts.get(risk, 0) + 1

    # Expected returns
    expected_returns = [
        rec.get("expected_return", 0)
        for rec in recommendations
        if rec.get("expected_return")
    ]
    avg_expected_return = (
        sum(expected_returns) / len(expected_returns) if expected_returns else 0
    )

    return {
        "total_allocation": total_allocation,
        "average_confidence": avg_confidence,
        "number_of_positions": len(recommendations),
        "risk_distribution": risk_counts,
        "average_expected_return": avg_expected_return,
    }


def save_analysis_results(
    results: Dict[str, Any], filename: Optional[str] = None
) -> str:
    """
    Save analysis results to a JSON file.

    Args:
        results: Analysis results dictionary
        filename: Optional filename (auto-generated if not provided)

    Returns:
        Path to saved file
    """
    if not filename:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        symbols = results.get("analysis_summary", {}).get(
            "symbols_analyzed", ["unknown"]
        )
        symbols_str = "_".join(symbols[:3])  # First 3 symbols
        filename = f"tradegraph_analysis_{symbols_str}_{timestamp}.json"

    # Ensure results directory exists
    os.makedirs("results", exist_ok=True)
    filepath = os.path.join("results", filename)

    try:
        with open(filepath, "w") as f:
            json.dump(results, f, indent=2, default=str)

        logger.info(f"Analysis results saved to {filepath}")
        return filepath

    except Exception as e:
        logger.error(f"Failed to save results: {str(e)}")
        raise


def load_analysis_results(filepath: str) -> Dict[str, Any]:
    """
    Load analysis results from a JSON file.

    Args:
        filepath: Path to the JSON file

    Returns:
        Analysis results dictionary
    """
    try:
        with open(filepath, "r") as f:
            results = json.load(f)

        logger.info(f"Analysis results loaded from {filepath}")
        return results

    except Exception as e:
        logger.error(f"Failed to load results from {filepath}: {str(e)}")
        raise


async def retry_async_operation(
    operation: Callable,
    max_retries: int = 3,
    delay: float = 1.0,
    exponential_backoff: bool = True,
) -> Any:
    """
    Retry an async operation with configurable backoff.

    Args:
        operation: Async function to retry
        max_retries: Maximum number of retry attempts
        delay: Initial delay between retries
        exponential_backoff: Whether to use exponential backoff

    Returns:
        Result of the operation
    """
    last_exception = None

    for attempt in range(max_retries + 1):
        try:
            return await operation()

        except Exception as e:
            last_exception = e

            if attempt == max_retries:
                logger.error(f"Operation failed after {max_retries} retries: {str(e)}")
                raise e

            wait_time = delay * (2**attempt) if exponential_backoff else delay
            logger.warning(
                f"Operation failed (attempt {attempt + 1}/{max_retries + 1}), retrying in {wait_time}s: {str(e)}"
            )
            await asyncio.sleep(wait_time)

    raise last_exception


def calculate_sharpe_ratio(returns: List[float], risk_free_rate: float = 0.02) -> float:
    """
    Calculate Sharpe ratio for a series of returns.

    Args:
        returns: List of returns
        risk_free_rate: Risk-free rate (default 2%)

    Returns:
        Sharpe ratio
    """
    if not returns or len(returns) < 2:
        return 0.0

    import numpy as np

    returns_array = np.array(returns)
    excess_returns = returns_array - risk_free_rate

    if np.std(excess_returns) == 0:
        return 0.0

    return np.mean(excess_returns) / np.std(excess_returns)


def calculate_max_drawdown(prices: List[float]) -> float:
    """
    Calculate maximum drawdown from a series of prices.

    Args:
        prices: List of prices

    Returns:
        Maximum drawdown as a percentage
    """
    if not prices or len(prices) < 2:
        return 0.0

    import numpy as np

    prices_array = np.array(prices)
    peak = np.maximum.accumulate(prices_array)
    drawdown = (prices_array - peak) / peak

    return float(np.min(drawdown))


def get_market_hours_status() -> Dict[str, Any]:
    """
    Check if markets are currently open (simplified US market hours).

    Returns:
        Dictionary with market status information
    """
    now = datetime.now()

    # Simplified market hours (9:30 AM - 4:00 PM ET, Monday-Friday)
    # This is a basic implementation and doesn't account for holidays

    weekday = now.weekday()  # 0 = Monday, 6 = Sunday
    hour = now.hour
    minute = now.minute

    is_weekend = weekday >= 5  # Saturday or Sunday

    market_open_time = 9.5  # 9:30 AM
    market_close_time = 16.0  # 4:00 PM
    current_time = hour + minute / 60.0

    is_market_hours = (
        not is_weekend and market_open_time <= current_time <= market_close_time
    )

    if is_weekend:
        status = "closed_weekend"
    elif current_time < market_open_time:
        status = "pre_market"
    elif current_time > market_close_time:
        status = "after_market"
    else:
        status = "open"

    return {
        "is_open": is_market_hours,
        "status": status,
        "current_time": now.isoformat(),
        "next_open": "calculated_separately",  # Would need more complex logic
        "timezone": "US/Eastern",  # Simplified
    }


def validate_analysis_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate and normalize analysis configuration.

    Args:
        config: Configuration dictionary

    Returns:
        Validated configuration
    """
    validated_config = config.copy()

    # Validate portfolio size
    portfolio_size = validated_config.get("portfolio_size", 100000)
    if portfolio_size <= 0:
        logger.warning("Invalid portfolio size, using default")
        validated_config["portfolio_size"] = 100000

    # Validate risk tolerance
    valid_risk_levels = ["conservative", "medium", "aggressive"]
    risk_tolerance = validated_config.get("risk_tolerance", "medium")
    if risk_tolerance not in valid_risk_levels:
        logger.warning(f"Invalid risk tolerance '{risk_tolerance}', using 'medium'")
        validated_config["risk_tolerance"] = "medium"

    # Validate time horizon
    valid_time_horizons = ["short_term", "medium_term", "long_term"]
    time_horizon = validated_config.get("time_horizon", "medium_term")
    if time_horizon not in valid_time_horizons:
        logger.warning(f"Invalid time horizon '{time_horizon}', using 'medium_term'")
        validated_config["time_horizon"] = "medium_term"

    # Validate symbols
    symbols = validated_config.get("symbols", [])
    if not symbols:
        raise ValueError("No symbols provided for analysis")

    validated_symbols = validate_symbols(symbols)
    if not validated_symbols:
        raise ValueError("No valid symbols found")

    validated_config["symbols"] = validated_symbols

    return validated_config


class PerformanceTimer:
    """Simple performance timer for measuring execution time."""

    def __init__(self, name: str = "operation"):
        self.name = name
        self.start_time = None
        self.end_time = None

    def __enter__(self):
        self.start_time = datetime.now()
        logger.debug(f"Starting {self.name}")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = datetime.now()
        duration = (self.end_time - self.start_time).total_seconds()
        logger.info(f"{self.name} completed in {duration:.2f} seconds")

    @property
    def duration(self) -> Optional[float]:
        if self.start_time and self.end_time:
            return (self.end_time - self.start_time).total_seconds()
        return None
