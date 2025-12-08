"""Utilities for working with asset symbols."""
from __future__ import annotations

from typing import Dict, List, Literal, TypedDict


AssetType = Literal["equity", "crypto"]


class SymbolResolution(TypedDict):
    """Normalized view of a requested asset symbol."""

    requested_symbol: str
    resolved_symbol: str
    base_symbol: str
    asset_type: AssetType


# Common crypto shortcuts we automatically normalize to Yahoo Finance tickers.
_CRYPTO_TICKER_MAP: Dict[str, str] = {
    "BTC": "BTC-USD",
    "BTC-USD": "BTC-USD",
    "BITCOIN": "BTC-USD",
    "ETH": "ETH-USD",
    "ETH-USD": "ETH-USD",
    "ETHEREUM": "ETH-USD",
    "SOL": "SOL-USD",
    "SOL-USD": "SOL-USD",
    "SOLANA": "SOL-USD",
    "ADA": "ADA-USD",
    "ADA-USD": "ADA-USD",
    "CARDANO": "ADA-USD",
    "XRP": "XRP-USD",
    "XRP-USD": "XRP-USD",
    "RIPPLE": "XRP-USD",
    "DOGE": "DOGE-USD",
    "DOGE-USD": "DOGE-USD",
    "DOGECOIN": "DOGE-USD",
    "LTC": "LTC-USD",
    "LTC-USD": "LTC-USD",
    "LITECOIN": "LTC-USD",
    "BNB": "BNB-USD",
    "BNB-USD": "BNB-USD",
    "AVAX": "AVAX-USD",
    "AVAX-USD": "AVAX-USD",
    "POLYGON": "MATIC-USD",
    "MATIC": "MATIC-USD",
    "MATIC-USD": "MATIC-USD",
    "DOT": "DOT-USD",
    "DOT-USD": "DOT-USD",
    "POLKADOT": "DOT-USD",
    "SHIB": "SHIB-USD",
    "SHIB-USD": "SHIB-USD",
}

_CRYPTO_BASES = {ticker.split("-")[0] for ticker in _CRYPTO_TICKER_MAP.values()}


def resolve_symbol(symbol: str) -> SymbolResolution:
    """Return a normalized representation of the requested symbol.

    Stocks pass through unchanged while common crypto shortcuts are mapped to the
    Yahoo Finance representation (e.g. ``BTC`` -> ``BTC-USD``).
    """

    requested = symbol.strip()
    cleaned = requested.upper().replace("CRYPTO:", "")
    cleaned = cleaned.replace(" ", "")

    if cleaned in _CRYPTO_TICKER_MAP:
        resolved = _CRYPTO_TICKER_MAP[cleaned]
        base = resolved.split("-")[0]
        return SymbolResolution(
            requested_symbol=requested,
            resolved_symbol=resolved,
            base_symbol=base,
            asset_type="crypto",
        )

    if cleaned.endswith("-USD") and cleaned[:-4] in _CRYPTO_BASES:
        base = cleaned[:-4]
        return SymbolResolution(
            requested_symbol=requested,
            resolved_symbol=cleaned,
            base_symbol=base,
            asset_type="crypto",
        )

    return SymbolResolution(
        requested_symbol=requested,
        resolved_symbol=cleaned,
        base_symbol=cleaned,
        asset_type="equity",
    )


def summarize_assets(symbols: List[str]) -> Dict[AssetType, int]:
    """Return a simple count of equities vs crypto assets."""

    summary: Dict[AssetType, int] = {"equity": 0, "crypto": 0}
    for symbol in symbols:
        resolution = resolve_symbol(symbol)
        summary[resolution["asset_type"]] += 1
    return summary
