from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime, timedelta, timezone
import aiohttp
import pandas as pd
from loguru import logger

from .base_agent import BaseAgent
from ..models.financial_data import CompanyFinancials, MarketData, TechnicalIndicators
from ..config.settings import settings
from ..services.market_data_clients import FinnhubClient, BinanceClient
from ..services.alpha_vantage_service import AlphaVantageClient


class FinancialAnalysisAgent(BaseAgent):
    def __init__(self, **kwargs):
        finnhub_client = kwargs.pop("finnhub_client", None)
        binance_client = kwargs.pop("binance_client", None)
        super().__init__(
            name="FinancialAnalysisAgent",
            description="Analyzes company financials and technical indicators",
            **kwargs,
        )
        self.session: Optional[aiohttp.ClientSession] = None
        self.finnhub_client = finnhub_client or FinnhubClient(settings.finnhub_api_key)
        self.binance_client = binance_client or BinanceClient()
        alpha_vantage_client = kwargs.pop("alpha_vantage_client", None)
        self.alpha_vantage_client = alpha_vantage_client
        self._owns_alpha_client = False
        if not self.alpha_vantage_client and settings.alpha_vantage_api_key:
            self.alpha_vantage_client = AlphaVantageClient(
                settings.alpha_vantage_api_key
            )
            self._owns_alpha_client = True
        self._owns_finnhub = finnhub_client is None
        self._owns_binance = binance_client is None
        self._profile_cache: Dict[str, Optional[Dict[str, Any]]] = {}

    async def start(self) -> None:
        await super().start()
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=settings.analysis_timeout_seconds)
        )

    async def stop(self) -> None:
        if self.session:
            await self.session.close()
        if self._owns_finnhub:
            await self.finnhub_client.close()
        if self._owns_binance:
            await self.binance_client.close()
        if self._owns_alpha_client and self.alpha_vantage_client:
            await self.alpha_vantage_client.close()
        await super().stop()

    async def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        symbols = input_data.get("symbols", [])
        include_financials = input_data.get("include_financials", True)
        include_technical = input_data.get("include_technical", True)
        include_market_data = input_data.get("include_market_data", True)
        alpha_requests = input_data.get("alpha_vantage_requests") or {}
        alpha_enabled = bool(
            self.alpha_vantage_client and alpha_requests.get("datasets")
        )

        logger.info(f"Analyzing financial data for symbols: {symbols}")

        results = {}
        alpha_summary: Dict[str, Any] = {}
        if alpha_enabled:
            alpha_summary = {
                "per_symbol": {},
                "global": await self._collect_alpha_global_data(alpha_requests),
                "requested_datasets": alpha_requests.get("datasets", []),
            }

        for symbol in symbols:
            try:
                symbol_data = {}
                market_data: Optional[MarketData] = None

                alpha_symbol_data: Dict[str, Any] = {}
                if alpha_enabled:
                    alpha_symbol_data = await self._collect_alpha_symbol_data(
                        symbol, alpha_requests
                    )

                if include_market_data:
                    if self._is_crypto(symbol):
                        market_data = await self._get_crypto_market_data(symbol)
                    else:
                        market_data = await self._get_equity_market_data(symbol)
                    symbol_data["market_data"] = (
                        market_data.model_dump() if market_data else None
                    )

                if include_financials:
                    if self._is_crypto(symbol):
                        symbol_data["financials"] = None
                    else:
                        financials = await self._get_company_financials(
                            symbol,
                            market_data,
                            alpha_symbol_data.get("fundamentals"),
                        )
                        symbol_data["financials"] = (
                            financials.model_dump() if financials else None
                        )

                if include_technical:
                    if self._is_crypto(symbol):
                        technical = await self._get_crypto_technical_indicators(symbol)
                    else:
                        technical = await self._get_equity_technical_indicators(symbol)
                    symbol_data["technical_indicators"] = (
                        technical.model_dump() if technical else None
                    )

                if alpha_symbol_data:
                    symbol_data["alpha_vantage"] = alpha_symbol_data
                    if alpha_summary:
                        alpha_summary.setdefault("per_symbol", {})[
                            symbol
                        ] = alpha_symbol_data

                results[symbol] = symbol_data

            except Exception as e:
                logger.error(f"Error analyzing {symbol}: {str(e)}")
                results[symbol] = {"error": str(e)}

        response: Dict[str, Any] = {
            "analysis_results": results,
            "analysis_timestamp": datetime.now().isoformat(),
        }
        if alpha_summary:
            response["alpha_vantage"] = alpha_summary
        return response

    async def _get_equity_market_data(self, symbol: str) -> Optional[MarketData]:
        try:
            quote = await self.finnhub_client.get_quote(symbol)
            if not quote:
                return None

            current_price = quote.get("c")
            open_price = quote.get("o")
            if current_price is None or open_price is None:
                return None

            change = float(current_price) - float(open_price)
            change_percent = (change / float(open_price)) * 100 if open_price else 0.0
            volume = int(quote.get("v") or 0)
            market_cap = await self._get_market_cap(symbol)

            return MarketData(
                symbol=symbol,
                current_price=float(current_price),
                change=float(change),
                change_percent=float(change_percent),
                volume=volume,
                market_cap=market_cap,
                pe_ratio=None,
                timestamp=datetime.now(),
            )

        except Exception as e:
            logger.error(f"Error fetching market data for {symbol}: {str(e)}")
            return None

    async def _get_crypto_market_data(self, symbol: str) -> Optional[MarketData]:
        try:
            klines = await self.binance_client.get_klines(
                symbol,
                interval="1m",
                limit=120,
            )
            if not klines:
                return None

            first = klines[0]
            last = klines[-1]
            start_price = float(first[1])
            end_price = float(last[4])
            change = end_price - start_price
            change_percent = (change / start_price * 100) if start_price else 0.0
            volume = sum(float(kline[5]) for kline in klines)

            return MarketData(
                symbol=symbol,
                current_price=end_price,
                change=change,
                change_percent=change_percent,
                volume=int(volume),
                market_cap=None,
                pe_ratio=None,
                timestamp=datetime.now(),
            )

        except Exception as e:
            logger.error(f"Error fetching crypto market data for {symbol}: {str(e)}")
            return None

    async def _get_company_financials(
        self,
        symbol: str,
        market_data: Optional[MarketData],
        alpha_fundamentals: Optional[Dict[str, Any]] = None,
    ) -> Optional[CompanyFinancials]:
        try:
            details = await self._get_company_profile(symbol)
            if not details and not market_data and not alpha_fundamentals:
                return None

            high_52, low_52 = await self._get_52_week_range(symbol)

            fundamentals = alpha_fundamentals or {}
            company_name = (
                fundamentals.get("name") or (details or {}).get("name") or symbol
            )
            market_cap = fundamentals.get("market_cap") or (details or {}).get(
                "market_cap"
            )
            fifty_two_week_high = fundamentals.get("fifty_two_week_high") or high_52
            fifty_two_week_low = fundamentals.get("fifty_two_week_low") or low_52
            financials = CompanyFinancials(
                symbol=symbol,
                company_name=company_name,
                market_cap=market_cap,
                current_price=market_data.current_price if market_data else None,
                fifty_two_week_high=fifty_two_week_high,
                fifty_two_week_low=fifty_two_week_low,
                pe_ratio=fundamentals.get("pe_ratio"),
                eps=fundamentals.get("eps"),
                revenue=fundamentals.get("revenue"),
                dividend_yield=fundamentals.get("dividend_yield"),
                return_on_equity=fundamentals.get("return_on_equity"),
                return_on_assets=fundamentals.get("return_on_assets"),
                debt_to_equity=fundamentals.get("debt_to_equity"),
                current_ratio=fundamentals.get("current_ratio"),
                price_to_book=fundamentals.get("price_to_book"),
                beta=fundamentals.get("beta"),
                report_date=datetime.now(),
                report_type="summary",
            )

            return financials

        except Exception as e:
            logger.error(f"Error fetching financials for {symbol}: {str(e)}")
            return None

    async def _get_equity_technical_indicators(
        self, symbol: str
    ) -> Optional[TechnicalIndicators]:
        try:
            now = datetime.now(timezone.utc)
            candles = await self.finnhub_client.get_candles(
                symbol,
                resolution="D",
                start=now - timedelta(days=160),
                end=now,
            )
            if candles.get("s") != "ok":
                return None

            closes = candles.get("c", [])
            highs = candles.get("h", [])
            lows = candles.get("l", [])
            if len(closes) < 50:
                return None

            close_prices = pd.Series([float(value) for value in closes])
            high_series = pd.Series([float(value) for value in highs])
            low_series = pd.Series([float(value) for value in lows])

            return self._build_technical_indicators(
                symbol, close_prices, high_series, low_series
            )

        except Exception as e:
            logger.error(
                f"Error calculating technical indicators for {symbol}: {str(e)}"
            )
            return None

    async def _get_crypto_technical_indicators(
        self, symbol: str
    ) -> Optional[TechnicalIndicators]:
        try:
            klines = await self.binance_client.get_klines(
                symbol,
                interval="1d",
                limit=160,
            )
            if len(klines) < 50:
                return None

            close_prices = pd.Series([float(item[4]) for item in klines])
            high_series = pd.Series([float(item[2]) for item in klines])
            low_series = pd.Series([float(item[3]) for item in klines])

            return self._build_technical_indicators(
                symbol, close_prices, high_series, low_series
            )

        except Exception as e:
            logger.error(
                f"Error calculating crypto technical indicators for {symbol}: {str(e)}"
            )
            return None

    async def _collect_alpha_symbol_data(
        self, symbol: str, requests: Dict[str, Any]
    ) -> Dict[str, Any]:
        if not self.alpha_vantage_client:
            return {}
        datasets = self._alpha_dataset_flags(requests)
        if not datasets:
            return {}

        payload: Dict[str, Any] = {}
        interval = requests.get("intraday_interval", "15min")
        technical_interval = (
            requests.get("technical_interval")
            or requests.get("intraday_interval")
            or "daily"
        )
        try:
            technical_period = int(requests.get("technical_time_period", 20))
        except (TypeError, ValueError):
            technical_period = 20
        series_type = requests.get("technical_series_type", "close")

        if "daily" in datasets:
            daily = await self.alpha_vantage_client.get_daily_time_series(symbol)
            if daily:
                payload["daily"] = daily

        if "intraday" in datasets:
            intraday = await self.alpha_vantage_client.get_intraday_time_series(
                symbol, interval=interval
            )
            if intraday:
                payload["intraday"] = intraday

        if "technical" in datasets:
            indicator_payload: Dict[str, Any] = {}
            raw_indicators = self._ensure_list(
                requests.get("technical_indicators")
            ) or ["SMA", "EMA", "RSI"]
            for indicator_name in raw_indicators:
                indicator_name = indicator_name.strip()
                if not indicator_name:
                    continue
                technical = await self.alpha_vantage_client.get_technical_indicator(
                    symbol,
                    indicator=indicator_name,
                    interval=technical_interval,
                    time_period=technical_period,
                    series_type=series_type,
                )
                if technical:
                    indicator_payload[indicator_name.upper()] = technical
            if indicator_payload:
                payload["technical_indicators"] = indicator_payload

        if "fundamentals" in datasets:
            fundamentals = await self.alpha_vantage_client.get_company_overview(symbol)
            if fundamentals:
                payload["fundamentals"] = fundamentals

        return payload

    async def _collect_alpha_global_data(
        self, requests: Dict[str, Any]
    ) -> Dict[str, Any]:
        if not self.alpha_vantage_client:
            return {}
        datasets = self._alpha_dataset_flags(requests)
        if not datasets:
            return {}

        payload: Dict[str, Any] = {}

        if "sector" in datasets:
            sector_performance = (
                await self.alpha_vantage_client.get_sector_performance()
            )
            if sector_performance:
                payload["sector_performance"] = sector_performance

        if "fx" in datasets:
            quotes = []
            fx_pairs = self._normalize_pairs(
                self._ensure_list(requests.get("fx_pairs")),
                fallback=["EUR/USD"],
            )
            for base, quote in fx_pairs:
                rate = await self.alpha_vantage_client.get_fx_rate(base, quote)
                if rate:
                    quotes.append(rate)
            if quotes:
                payload["fx_quotes"] = quotes

        if "crypto" in datasets:
            crypto_quotes = []
            crypto_pairs = self._normalize_pairs(
                self._ensure_list(requests.get("crypto_pairs")),
                fallback=["BTC/USD"],
            )
            for base, quote in crypto_pairs:
                rate = await self.alpha_vantage_client.get_crypto_rate(base, quote)
                if rate:
                    crypto_quotes.append(rate)
            if crypto_quotes:
                payload["crypto_quotes"] = crypto_quotes

        return {key: value for key, value in payload.items() if value}

    def _alpha_dataset_flags(self, requests: Dict[str, Any]) -> List[str]:
        datasets = requests.get("datasets") or []
        normalized = []
        for item in datasets:
            try:
                normalized_value = str(item).strip().lower()
            except Exception:  # pragma: no cover - defensive
                continue
            if normalized_value:
                normalized.append(normalized_value)
        return normalized

    def _normalize_pairs(
        self, pairs: List[str], fallback: Optional[List[str]] = None
    ) -> List[Tuple[str, str]]:
        normalized: List[Tuple[str, str]] = []
        for pair in pairs:
            base, quote = self._split_pair(pair)
            if base and quote:
                normalized.append((base, quote))
        if not normalized and fallback:
            for pair in fallback:
                base, quote = self._split_pair(pair)
                if base and quote:
                    normalized.append((base, quote))
        return normalized

    def _ensure_list(self, value: Any) -> List[str]:
        if value is None:
            return []
        if isinstance(value, str):
            return [item.strip() for item in value.split(",") if item.strip()]
        if isinstance(value, (list, tuple, set)):
            return [
                str(item).strip()
                for item in value
                if isinstance(item, (str, int, float)) and str(item).strip()
            ]
        return []

    @staticmethod
    def _split_pair(value: str) -> Tuple[Optional[str], Optional[str]]:
        if not value:
            return None, None
        cleaned = value.replace("-", "/").replace(" ", "")
        if "/" in cleaned:
            base, quote = cleaned.split("/", 1)
        else:
            midpoint = len(cleaned) // 2
            base, quote = cleaned[:midpoint], cleaned[midpoint:]
        base = base.strip().upper()
        quote = quote.strip().upper()
        if not base or not quote:
            return None, None
        return base, quote

    def _build_technical_indicators(
        self,
        symbol: str,
        close_prices: pd.Series,
        high_series: pd.Series,
        low_series: pd.Series,
    ) -> TechnicalIndicators:
        sma_20 = close_prices.rolling(window=20).mean().iloc[-1]
        sma_50 = close_prices.rolling(window=50).mean().iloc[-1]

        ema_12_series = close_prices.ewm(span=12).mean()
        ema_26_series = close_prices.ewm(span=26).mean()
        ema_12 = ema_12_series.iloc[-1]
        ema_26 = ema_26_series.iloc[-1]

        delta = close_prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs)).iloc[-1]

        macd_line = ema_12_series - ema_26_series
        macd_signal = macd_line.ewm(span=9).mean().iloc[-1]

        bb_window = 20
        bb_std = close_prices.rolling(window=bb_window).std().iloc[-1]
        bb_sma = close_prices.rolling(window=bb_window).mean().iloc[-1]
        bollinger_upper = bb_sma + (bb_std * 2)
        bollinger_lower = bb_sma - (bb_std * 2)

        recent_high = float(high_series.tail(20).max())
        recent_low = float(low_series.tail(20).min())

        return TechnicalIndicators(
            symbol=symbol,
            sma_20=float(sma_20) if pd.notna(sma_20) else None,
            sma_50=float(sma_50) if pd.notna(sma_50) else None,
            ema_12=float(ema_12) if pd.notna(ema_12) else None,
            ema_26=float(ema_26) if pd.notna(ema_26) else None,
            rsi=float(rsi) if pd.notna(rsi) else None,
            macd=float(macd_line.iloc[-1]) if pd.notna(macd_line.iloc[-1]) else None,
            macd_signal=float(macd_signal) if pd.notna(macd_signal) else None,
            bollinger_upper=float(bollinger_upper)
            if pd.notna(bollinger_upper)
            else None,
            bollinger_lower=float(bollinger_lower)
            if pd.notna(bollinger_lower)
            else None,
            support_level=recent_low,
            resistance_level=recent_high,
            timestamp=datetime.now(),
        )

    async def _health_check_impl(self) -> None:
        try:
            quote = await self.finnhub_client.get_quote("AAPL")
            if not quote or quote.get("c") is None:
                raise Exception("Finnhub quote unavailable")
        except Exception as e:
            raise Exception(f"Finnhub health check failed: {str(e)}")

    async def _get_market_cap(self, symbol: str) -> Optional[float]:
        profile = await self._get_company_profile(symbol)
        return (profile or {}).get("market_cap")

    async def _get_company_profile(self, symbol: str) -> Optional[Dict[str, Any]]:
        if symbol in self._profile_cache:
            return self._profile_cache[symbol]
        profile = await self.finnhub_client.get_company_profile(symbol)
        if profile:
            normalized = {
                "name": profile.get("name") or profile.get("ticker") or symbol,
                "market_cap": profile.get("marketCapitalization"),
            }
        else:
            normalized = None
        self._profile_cache[symbol] = normalized
        return normalized

    async def _get_52_week_range(
        self, symbol: str
    ) -> Tuple[Optional[float], Optional[float]]:
        now = datetime.now(timezone.utc)
        try:
            candles = await self.finnhub_client.get_candles(
                symbol,
                resolution="D",
                start=now - timedelta(days=365),
                end=now,
            )
        except Exception as exc:
            logger.warning(f"Failed to compute 52-week range for {symbol}: {exc}")
            return None, None
        if candles.get("s") != "ok":
            return None, None
        highs = candles.get("h", [])
        lows = candles.get("l", [])
        if not highs or not lows:
            return None, None
        return float(max(highs)), float(min(lows))

    @staticmethod
    def _is_crypto(symbol: str) -> bool:
        normalized = symbol.upper()
        if normalized.startswith("X:") or normalized.startswith("CRYPTO:"):
            return True
        if "-" in normalized:
            _, suffix = normalized.split("-", 1)
            return suffix in {"USD", "USDT", "BTC", "ETH"}
        return False
