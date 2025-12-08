from typing import Optional

from fastapi import APIRouter, HTTPException
from tradegraph_financial_advisor.services.db_manager import db_manager
from ..models import APIResponse

router = APIRouter()

@router.get("/data", response_model=APIResponse)
async def get_graph_data():
    """Get knowledge graph data for visualization."""
    try:
        data = db_manager.get_graph_data()
        return APIResponse(
            success=True,
            data=data,
            message="Graph data retrieved successfully"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/stock/{symbol}/history", response_model=APIResponse)
async def get_stock_history(symbol: str):
    """Get historical stock data from DuckDB."""
    try:
        data = db_manager.get_stock_history(symbol)
        return APIResponse(
            success=True,
            data=data,
            message=f"Historical data for {symbol} retrieved successfully"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/articles", response_model=APIResponse)
async def get_recent_articles(limit: int = 10, symbols: Optional[str] = None):
    """Return recent articles stored in the knowledge graph."""
    try:
        symbol_list = None
        if symbols:
            symbol_list = [item.strip().upper() for item in symbols.split(",") if item.strip()]
        data = db_manager.get_recent_articles(symbol_list, limit)
        return APIResponse(
            success=True,
            data=data,
            message="Knowledge graph articles retrieved successfully",
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/queries", response_model=APIResponse)
async def get_recent_queries(limit: int = 10):
    """Return recent analysis queries stored in DuckDB."""
    try:
        data = db_manager.get_recent_queries(limit)
        return APIResponse(
            success=True,
            data=data,
            message="Recent queries retrieved successfully",
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
