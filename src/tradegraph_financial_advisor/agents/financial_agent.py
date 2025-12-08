from typing import Any, Dict, Optional
from datetime import datetime
import aiohttp
import yfinance as yf
import pandas as pd
from loguru import logger

from .base_agent import BaseAgent
from ..models.financial_data import CompanyFinancials, MarketData, TechnicalIndicators
from ..config.settings import settings


class FinancialAnalysisAgent(BaseAgent):
    def __init__(self, **kwargs):
        super().__init__(
            name="FinancialAnalysisAgent",
            description="Analyzes company financials and technical indicators",
            **kwargs
        )
        self.session: Optional[aiohttp.ClientSession] = None

    async def start(self) -> None:
        await super().start()
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=settings.analysis_timeout_seconds)
        )

    async def stop(self) -> None:
        if self.session:
            await self.session.close()
        await super().stop()

    async def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        symbols = input_data.get("symbols", [])
        include_financials = input_data.get("include_financials", True)
        include_technical = input_data.get("include_technical", True)
        include_market_data = input_data.get("include_market_data", True)

        logger.info(f"Analyzing financial data for symbols: {symbols}")

        results = {}

        for symbol in symbols:
            try:
                symbol_data = {}

                if include_market_data:
                    market_data = await self._get_market_data(symbol)
                    symbol_data["market_data"] = market_data.dict() if market_data else None

                if include_financials:
                    financials = await self._get_company_financials(symbol)
                    symbol_data["financials"] = financials.dict() if financials else None

                if include_technical:
                    technical = await self._get_technical_indicators(symbol)
                    symbol_data["technical_indicators"] = technical.dict() if technical else None

                results[symbol] = symbol_data

            except Exception as e:
                logger.error(f"Error analyzing {symbol}: {str(e)}")
                results[symbol] = {"error": str(e)}

        return {
            "analysis_results": results,
            "analysis_timestamp": datetime.now().isoformat()
        }

    async def _get_market_data(self, symbol: str) -> Optional[MarketData]:
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            history = ticker.history(period="1d")

            if history.empty:
                return None

            latest = history.iloc[-1]

            market_data = MarketData(
                symbol=symbol,
                current_price=float(latest['Close']),
                change=float(latest['Close'] - latest['Open']),
                change_percent=float((latest['Close'] - latest['Open']) / latest['Open'] * 100),
                volume=int(latest['Volume']),
                market_cap=info.get('marketCap'),
                pe_ratio=info.get('trailingPE'),
                timestamp=datetime.now()
            )

            return market_data

        except Exception as e:
            logger.error(f"Error fetching market data for {symbol}: {str(e)}")
            return None

    async def _get_company_financials(self, symbol: str) -> Optional[CompanyFinancials]:
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info

            financials = CompanyFinancials(
                symbol=symbol,
                company_name=info.get('longName', symbol),
                market_cap=info.get('marketCap'),
                pe_ratio=info.get('trailingPE'),
                eps=info.get('trailingEps'),
                revenue=info.get('totalRevenue'),
                net_income=info.get('netIncomeToCommon'),
                debt_to_equity=info.get('debtToEquity'),
                current_ratio=info.get('currentRatio'),
                return_on_equity=info.get('returnOnEquity'),
                return_on_assets=info.get('returnOnAssets'),
                price_to_book=info.get('priceToBook'),
                dividend_yield=info.get('dividendYield'),
                beta=info.get('beta'),
                fifty_two_week_high=info.get('fiftyTwoWeekHigh'),
                fifty_two_week_low=info.get('fiftyTwoWeekLow'),
                current_price=info.get('currentPrice'),
                report_date=datetime.now(),
                report_type="quarterly"
            )

            return financials

        except Exception as e:
            logger.error(f"Error fetching financials for {symbol}: {str(e)}")
            return None

    async def _get_technical_indicators(self, symbol: str) -> Optional[TechnicalIndicators]:
        try:
            ticker = yf.Ticker(symbol)
            history = ticker.history(period="3mo")  # 3 months of data

            if len(history) < 50:  # Need enough data for indicators
                return None

            # Calculate technical indicators
            close_prices = history['Close']

            # Simple Moving Averages
            sma_20 = close_prices.rolling(window=20).mean().iloc[-1]
            sma_50 = close_prices.rolling(window=50).mean().iloc[-1]

            # Exponential Moving Averages
            ema_12_series = close_prices.ewm(span=12).mean()
            ema_26_series = close_prices.ewm(span=26).mean()
            ema_12 = ema_12_series.iloc[-1]
            ema_26 = ema_26_series.iloc[-1]

            # RSI (simplified calculation)
            delta = close_prices.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs)).iloc[-1]

            # MACD
            macd_line = ema_12_series - ema_26_series
            macd_signal = macd_line.ewm(span=9).mean().iloc[-1]

            # Bollinger Bands
            bb_window = 20
            bb_std = close_prices.rolling(window=bb_window).std().iloc[-1]
            bb_sma = close_prices.rolling(window=bb_window).mean().iloc[-1]
            bollinger_upper = bb_sma + (bb_std * 2)
            bollinger_lower = bb_sma - (bb_std * 2)

            # Support and Resistance (simplified)
            recent_high = history['High'].tail(20).max()
            recent_low = history['Low'].tail(20).min()

            technical = TechnicalIndicators(
                symbol=symbol,
                sma_20=float(sma_20) if not pd.isna(sma_20) else None,
                sma_50=float(sma_50) if not pd.isna(sma_50) else None,
                ema_12=float(ema_12) if not pd.isna(ema_12) else None,
                ema_26=float(ema_26) if not pd.isna(ema_26) else None,
                rsi=float(rsi) if not pd.isna(rsi) else None,
                macd=float(macd_line.iloc[-1]) if not pd.isna(macd_line.iloc[-1]) else None,
                macd_signal=float(macd_signal) if not pd.isna(macd_signal) else None,
                bollinger_upper=float(bollinger_upper) if not pd.isna(bollinger_upper) else None,
                bollinger_lower=float(bollinger_lower) if not pd.isna(bollinger_lower) else None,
                support_level=float(recent_low),
                resistance_level=float(recent_high),
                timestamp=datetime.now()
            )

            return technical

        except Exception as e:
            logger.error(f"Error calculating technical indicators for {symbol}: {str(e)}")
            return None

    async def _health_check_impl(self) -> None:
        # Test yfinance by fetching a simple stock quote
        try:
            ticker = yf.Ticker("AAPL")
            info = ticker.info
            if not info:
                raise Exception("Unable to fetch test data")
        except Exception as e:
            raise Exception(f"yfinance health check failed: {str(e)}")
