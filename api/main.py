import logging
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Dict, Any

from fastapi import (
    FastAPI,
    HTTPException,
    WebSocket,
    WebSocketDisconnect,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
import uvicorn

from .routers import analysis, portfolio, alerts, health, graph
from .middleware.rate_limiter import RateLimitMiddleware
from .middleware.logging import LoggingMiddleware
from .models import APIResponse
from .websocket_manager import WebSocketManager
from tradegraph_financial_advisor.services.db_manager import db_manager

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Global state for background tasks and WebSocket connections
websocket_manager = WebSocketManager()
background_tasks_status: Dict[str, Dict[str, Any]] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan context manager."""
    logger.info("🚀 TradeGraph Financial Advisor API starting up...")

    # Startup tasks
    try:
        # Initialize any required services
        # DBManager is initialized on import/first use, but we can verify here
        if db_manager.duckdb_conn:
            logger.info("✅ DuckDB connection active")
        if db_manager.neo4j_driver:
            logger.info("✅ Neo4j connection active")

        logger.info("✅ API services initialized")
        yield
    finally:
        # Cleanup tasks
        logger.info("🔄 API shutting down...")
        db_manager.close()
        await websocket_manager.disconnect_all()
        logger.info("✅ API shutdown complete")


# Create FastAPI application
app = FastAPI(
    title="TradeGraph Financial Advisor API",
    description="""
    🚀 **TradeGraph Financial Advisor API**

    A sophisticated multi-agent financial analysis system that provides:

    - **Real-time Financial Analysis**: Market data, news sentiment, and technical indicators
    - **AI-Powered Recommendations**: Buy/Sell/Hold recommendations with confidence scores
    - **Portfolio Optimization**: Risk-adjusted portfolio construction
    - **SEC Filing Analysis**: Deep analysis of 10-K and 10-Q reports
    - **Real-time Alerts**: WebSocket-based live updates

    ## Features

    - 🤖 **Multi-Agent Architecture**: Coordinated AI agents for comprehensive analysis
    - 📊 **Advanced Analytics**: Technical analysis, sentiment analysis, and fundamental analysis
    - 🔄 **Real-time Updates**: Live market data and analysis results
    - 🛡️ **Risk Management**: Sophisticated risk assessment and position sizing
    - 📈 **Portfolio Management**: Intelligent diversification and rebalancing

    ## Authentication

    Most endpoints require authentication. Use the `/auth/login` endpoint to obtain a JWT token.
    """,
    version="1.0.0",
    contact={
        "name": "TradeGraph Team",
        "url": "https://tradegraph.com",
        "email": "support@tradegraph.com",
    },
    license_info={
        "name": "MIT",
        "url": "https://opensource.org/licenses/MIT",
    },
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# Add middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(GZipMiddleware, minimum_size=1000)
app.add_middleware(RateLimitMiddleware, calls=100, period=60)  # 100 calls per minute
app.add_middleware(LoggingMiddleware)

# Include routers
app.include_router(health.router, prefix="/health", tags=["Health"])
app.include_router(analysis.router, prefix="/analysis", tags=["Analysis"])
app.include_router(portfolio.router, prefix="/portfolio", tags=["Portfolio"])
app.include_router(alerts.router, prefix="/alerts", tags=["Alerts"])
app.include_router(graph.router, prefix="/graph", tags=["Knowledge Graph"])

# Mount static files for frontend
try:
    app.mount("/static", StaticFiles(directory="frontend/dist"), name="static")
except RuntimeError:
    logger.warning("Frontend static files not found - running in API-only mode")


@app.get("/", response_class=HTMLResponse)
async def root():
    """Root endpoint serving the frontend or API information."""
    try:
        with open("frontend/dist/index.html", "r") as f:
            return f.read()
    except FileNotFoundError:
        # Fallback to basic API info if frontend not found
        return """
        <html>
            <head>
                <title>TradeGraph Financial Advisor API</title>
                <style>
                    body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
                           margin: 0; padding: 40px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                           color: white; min-height: 100vh; text-align: center; }
                    .container { max-width: 600px; margin: 0 auto; }
                    .logo { font-size: 3em; margin-bottom: 20px; }
                    .links { display: flex; gap: 20px; justify-content: center; margin-top: 40px; }
                    .link { background: rgba(255,255,255,0.2); padding: 15px 30px; border-radius: 10px;
                            text-decoration: none; color: white; transition: all 0.3s; }
                    .link:hover { background: rgba(255,255,255,0.3); transform: translateY(-2px); }
                </style>
            </head>
            <body>
                <div class="container">
                    <div class="logo">📈 TradeGraph API</div>
                    <p>Frontend not found - running in API-only mode</p>
                    <div class="links">
                        <a href="/docs" class="link">📚 API Documentation</a>
                        <a href="/redoc" class="link">📖 ReDoc</a>
                        <a href="/health" class="link">💚 Health Check</a>
                    </div>
                </div>
            </body>
        </html>
        """


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time updates."""
    await websocket_manager.connect(websocket)
    try:
        while True:
            # Keep connection alive and handle incoming messages
            data = await websocket.receive_text()
            logger.info(f"Received WebSocket message: {data}")

            # Echo back for now - in production, this would handle different message types
            await websocket_manager.send_personal_message(f"Echo: {data}", websocket)
    except WebSocketDisconnect:
        websocket_manager.disconnect(websocket)


@app.websocket("/ws/analysis/{analysis_id}")
async def analysis_websocket(websocket: WebSocket, analysis_id: str):
    """WebSocket endpoint for specific analysis updates."""
    await websocket_manager.connect(websocket, f"analysis_{analysis_id}")
    try:
        while True:
            data = await websocket.receive_text()
            # Handle analysis-specific WebSocket messages
            await websocket_manager.send_to_group(
                f"analysis_{analysis_id}",
                {
                    "type": "analysis_update",
                    "analysis_id": analysis_id,
                    "message": data,
                    "timestamp": datetime.now().isoformat(),
                },
            )
    except WebSocketDisconnect:
        websocket_manager.disconnect(websocket)


@app.get("/api/info")
async def api_info():
    """Get API information and status."""
    return APIResponse(
        success=True,
        data={
            "name": "TradeGraph Financial Advisor API",
            "version": "1.0.0",
            "description": "AI-powered financial analysis and trading recommendations",
            "features": [
                "Multi-agent financial analysis",
                "Real-time market data",
                "AI-powered recommendations",
                "Portfolio optimization",
                "Risk management",
                "WebSocket support",
            ],
            "endpoints": {
                "health": "/health",
                "analysis": "/analysis",
                "portfolio": "/portfolio",
                "alerts": "/alerts",
                "websocket": "/ws",
                "docs": "/docs",
            },
            "status": "operational",
            "uptime": "calculated_dynamically",
        },
        message="API is operational",
    )


# Error handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Handle HTTP exceptions."""
    payload = APIResponse(
        success=False,
        data=None,
        message=exc.detail,
        error={
            "type": "HTTPException",
            "status_code": exc.status_code,
            "detail": exc.detail,
        },
    )
    return JSONResponse(status_code=exc.status_code, content=payload.dict())


@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """Handle general exceptions."""
    logger.error(f"Unhandled exception: {str(exc)}", exc_info=True)
    payload = APIResponse(
        success=False,
        data=None,
        message="Internal server error",
        error={"type": "InternalServerError", "detail": "An unexpected error occurred"},
    )
    return JSONResponse(status_code=500, content=payload.dict())


# Background task management
@app.get("/api/tasks")
async def get_background_tasks():
    """Get status of background tasks."""
    return APIResponse(
        success=True,
        data={
            "active_tasks": len(background_tasks_status),
            "tasks": background_tasks_status,
        },
        message="Background tasks status retrieved",
    )


@app.get("/api/tasks/{task_id}")
async def get_task_status(task_id: str):
    """Get specific task status."""
    if task_id not in background_tasks_status:
        raise HTTPException(status_code=404, detail="Task not found")

    return APIResponse(
        success=True,
        data=background_tasks_status[task_id],
        message=f"Task {task_id} status retrieved",
    )


# Development server
if __name__ == "__main__":
    uvicorn.run(
        "api.main:app", host="0.0.0.0", port=8000, reload=True, log_level="info"
    )
