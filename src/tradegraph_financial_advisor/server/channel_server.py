"""FastAPI server that exposes financial channels over WebSockets."""

from __future__ import annotations

import asyncio
from typing import List, Optional

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Query, HTTPException
from fastapi.responses import JSONResponse
from loguru import logger

from ..services.channel_stream_service import (
    FinancialNewsChannelService,
    ChannelType,
    CHANNEL_REGISTRY,
)

app = FastAPI(
    title="TradeGraph Financial Channels",
    description="Real-time websocket feeds for multi-source news and pricing streams",
    version="1.0.0",
)

channel_service = FinancialNewsChannelService()


def _parse_symbols(symbols: Optional[str]) -> List[str]:
    if not symbols:
        return []
    return [symbol.strip() for symbol in symbols.split(",") if symbol.strip()]


@app.on_event("shutdown")
async def shutdown_event() -> None:
    await channel_service.close()


@app.get("/health")
async def health() -> JSONResponse:
    return JSONResponse(
        {
            "status": "ok",
            "channel_count": len(ChannelType),
        }
    )


@app.get("/channels")
async def list_channels() -> List[dict]:
    return channel_service.describe_channels()


@app.get("/channels/{channel_id}")
async def channel_snapshot(channel_id: str, symbols: Optional[str] = Query(None)):
    try:
        payload = await channel_service.fetch_channel_payload(
            channel_id, _parse_symbols(symbols)
        )
        return payload
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.websocket("/ws/{channel_id}")
async def channel_stream(websocket: WebSocket, channel_id: str) -> None:
    try:
        channel_type = ChannelType.from_value(channel_id)
    except ValueError as exc:  # pragma: no cover - validated per connection
        await websocket.close(code=4404, reason=str(exc))
        return

    definition = CHANNEL_REGISTRY[channel_type]
    await websocket.accept()

    symbols_param = websocket.query_params.get("symbols")
    symbols = _parse_symbols(symbols_param)

    try:
        while True:
            payload = await channel_service.fetch_channel_payload(
                channel_type.value, symbols or None
            )
            await websocket.send_json(payload)
            await asyncio.sleep(max(5, definition.refresh_seconds))
    except WebSocketDisconnect:
        logger.info("Websocket client disconnected: {}", channel_id)
    except Exception as exc:
        logger.error(f"Websocket streaming error for {channel_id}: {exc}")
        await websocket.close(code=1011, reason=str(exc))


__all__ = ["app"]
