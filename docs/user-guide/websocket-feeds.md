# WebSocket Market Data Feeds

TradeGraph now supports low-latency market data by connecting to real-time vendor feeds. The financial analysis agent uses these feeds automatically when `include_market_data=True`, so you can mix fresh tick data with Yahoo Finance fundamentals in a single run.

## Supported Providers

| Provider | Asset Class | Notes |
| --- | --- | --- |
| Polygon | Equities | Requires `POLYGON_API_KEY`. Uses trades channel `T.*`. |
| Alpaca | Equities | Requires `ALPACA_API_KEY` and `ALPACA_API_SECRET`. Supports `ALPACA_DATA_FEED` (`iex` or `sip`). |
| Binance | Crypto | No authentication required. Streams trades via `symbol@trade`. |

Use the environment variables (or settings) to select defaults:

```env
DEFAULT_EQUITY_FEED_PROVIDER=polygon
DEFAULT_CRYPTO_FEED_PROVIDER=binance
POLYGON_API_KEY=...
ALPACA_API_KEY=...
ALPACA_API_SECRET=...
ALPACA_DATA_FEED=iex
WEBSOCKET_TIMEOUT_SECONDS=10
```

You can override feeds per symbol at runtime with `asset_types` and `feed_overrides` fields on the `FinancialAnalysisAgent` input payload.

## Monitoring Latency

Each `MarketData` entry now reports two additional fields:

```
provider: Source of the trade (`polygon`, `alpaca`, `binance`, ...)
feed_latency_ms: Milliseconds between trade timestamp and ingestion
```

This helps dashboards surface stale feeds or connectivity issues.

## Quick CLI Test

The repository ships with `examples/websocket_feed_demo.py` for manual testing.

```bash
python examples/websocket_feed_demo.py AAPL --asset-type equity \
    --polygon-api-key $POLYGON_API_KEY
```

Specify `--provider alpaca` or `--asset-type crypto` to route to other feeds. The script prints the decoded trade payload along with the calculated latency so you can verify credentials before running full analyses.
```
