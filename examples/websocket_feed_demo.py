"""Quick script to test streaming market data feeds."""

import argparse
import asyncio
import os
from pprint import pprint

from tradegraph_financial_advisor.services.market_data import (
    MarketDataFeedConfig,
    MarketDataWebSocketClient,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Test WebSocket market feeds")
    parser.add_argument("symbol", help="Ticker symbol, e.g., AAPL or BTCUSDT")
    parser.add_argument(
        "--asset-type",
        default="equity",
        choices=["equity", "crypto"],
        help="Asset class for routing to the right feed",
    )
    parser.add_argument(
        "--provider",
        default=None,
        help="Override provider (binance, polygon, alpaca)",
    )
    parser.add_argument(
        "--equity-provider",
        default=os.getenv("DEFAULT_EQUITY_FEED_PROVIDER", "polygon"),
        help="Default provider for equities",
    )
    parser.add_argument(
        "--crypto-provider",
        default=os.getenv("DEFAULT_CRYPTO_FEED_PROVIDER", "binance"),
        help="Default provider for crypto symbols",
    )
    parser.add_argument(
        "--polygon-api-key",
        default=os.getenv("POLYGON_API_KEY"),
        help="Polygon API key",
    )
    parser.add_argument(
        "--alpaca-api-key",
        default=os.getenv("ALPACA_API_KEY"),
        help="Alpaca API key",
    )
    parser.add_argument(
        "--alpaca-api-secret",
        default=os.getenv("ALPACA_API_SECRET"),
        help="Alpaca API secret",
    )
    parser.add_argument(
        "--alpaca-feed",
        default=os.getenv("ALPACA_DATA_FEED", "iex"),
        help="Alpaca feed (iex or sip)",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=int(os.getenv("WEBSOCKET_TIMEOUT_SECONDS", "10")),
        help="Seconds to wait for a trade message",
    )
    return parser.parse_args()


def build_config(args: argparse.Namespace) -> MarketDataFeedConfig:
    return MarketDataFeedConfig(
        equity_provider=args.equity_provider,
        crypto_provider=args.crypto_provider,
        polygon_api_key=args.polygon_api_key,
        alpaca_api_key=args.alpaca_api_key,
        alpaca_api_secret=args.alpaca_api_secret,
        alpaca_feed=args.alpaca_feed,
        timeout_seconds=args.timeout,
    )


async def main() -> None:
    args = parse_args()
    config = build_config(args)

    async with MarketDataWebSocketClient(config=config) as client:
        trade = await client.get_realtime_trade(
            args.symbol,
            asset_type=args.asset_type,
            provider_override=args.provider,
        )

    pprint(
        {
            "symbol": trade.symbol,
            "price": trade.price,
            "size": trade.size,
            "provider": trade.provider,
            "timestamp": trade.timestamp.isoformat(),
            "latency_ms": trade.latency_ms,
        }
    )


if __name__ == "__main__":
    asyncio.run(main())
