import asyncio
import uuid
from datetime import datetime
from typing import Dict, Optional

from fastapi import APIRouter, HTTPException, BackgroundTasks, Depends
from fastapi.responses import StreamingResponse

from ..models import (
    APIResponse,
    AnalysisRequest,
    QuickAnalysisRequest,
    AnalysisResult,
    AnalysisStatus,
)

from ..auth.dependencies import get_current_user
from ..websocket_manager import WebSocketManager
from radgegraph_financial_advisor import FinancialAdvisor

router = APIRouter()

# Global instances
financial_advisor = FinancialAdvisor()
websocket_manager = WebSocketManager()

# Store analysis results temporarily (in production, use a database)
analysis_store: Dict[str, AnalysisResult] = {}


async def run_analysis_task(
    analysis_id: str, request: AnalysisRequest, websocket_manager: WebSocketManager
):
    """Background task to run financial analysis."""
    try:
        # Update status to in_progress
        analysis_store[analysis_id].status = AnalysisStatus.IN_PROGRESS
        analysis_store[analysis_id].progress = 0.0

        await websocket_manager.send_analysis_update(
            analysis_id,
            {
                "status": "in_progress",
                "progress": 0.0,
                "message": "Starting analysis...",
            },
        )

        # Run the analysis
        await websocket_manager.send_analysis_update(
            analysis_id, {"progress": 20.0, "message": "Collecting market data..."}
        )

        results = await financial_advisor.analyze_portfolio(
            symbols=request.symbols,
            portfolio_size=request.portfolio_size,
            risk_tolerance=request.risk_tolerance,
            time_horizon=request.time_horizon,
            include_reports=request.include_reports,
        )

        await websocket_manager.send_analysis_update(
            analysis_id, {"progress": 80.0, "message": "Generating recommendations..."}
        )

        # Store results
        analysis_store[analysis_id].status = AnalysisStatus.COMPLETED
        analysis_store[analysis_id].completed_at = datetime.now()
        analysis_store[analysis_id].results = results
        analysis_store[analysis_id].progress = 100.0

        await websocket_manager.send_analysis_update(
            analysis_id,
            {
                "status": "completed",
                "progress": 100.0,
                "message": "Analysis completed successfully",
                "results": results,
            },
        )

    except Exception as e:
        # Update status to failed
        analysis_store[analysis_id].status = AnalysisStatus.FAILED
        analysis_store[analysis_id].error_message = str(e)
        analysis_store[analysis_id].completed_at = datetime.now()

        await websocket_manager.send_analysis_update(
            analysis_id, {"status": "failed", "message": f"Analysis failed: {str(e)}"}
        )


@router.post("/comprehensive", response_model=APIResponse)
async def start_comprehensive_analysis(
    request: AnalysisRequest,
    background_tasks: BackgroundTasks,
    current_user: Optional[Dict] = Depends(get_current_user),
):
    """
    Start a comprehensive financial analysis.

    This endpoint initiates a background analysis task that includes:
    - Multi-source news analysis
    - Technical indicator calculation
    - Fundamental analysis
    - Optional SEC filing analysis
    - Portfolio optimization
    - Risk assessment

    Returns an analysis ID that can be used to track progress via WebSocket or polling.
    """
    try:
        # Generate unique analysis ID
        analysis_id = str(uuid.uuid4())

        # Create analysis record
        analysis_record = AnalysisResult(
            analysis_id=analysis_id,
            status=AnalysisStatus.PENDING,
            symbols=request.symbols,
            created_at=datetime.now(),
        )

        analysis_store[analysis_id] = analysis_record

        # Start background task
        background_tasks.add_task(
            run_analysis_task, analysis_id, request, websocket_manager
        )

        return APIResponse(
            success=True,
            data={
                "analysis_id": analysis_id,
                "status": "pending",
                "symbols": request.symbols,
                "estimated_completion": "2-5 minutes",
                "websocket_url": f"/ws/analysis/{analysis_id}",
                "polling_url": f"/analysis/status/{analysis_id}",
            },
            message="Comprehensive analysis started successfully",
        )

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to start analysis: {str(e)}"
        )


@router.post("/quick", response_model=APIResponse)
async def quick_analysis(
    request: QuickAnalysisRequest,
    current_user: Optional[Dict] = Depends(get_current_user),
):
    """
    Perform quick financial analysis.

    This is a faster analysis that provides basic recommendations without
    deep SEC filing analysis or extensive backtesting.
    """
    try:
        results = await financial_advisor.quick_analysis(
            symbols=request.symbols, analysis_type=request.analysis_type
        )

        return APIResponse(
            success=True, data=results, message="Quick analysis completed successfully"
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Quick analysis failed: {str(e)}")


@router.get("/status/{analysis_id}", response_model=APIResponse)
async def get_analysis_status(analysis_id: str):
    """Get the status of an analysis by ID."""
    if analysis_id not in analysis_store:
        raise HTTPException(status_code=404, detail="Analysis not found")

    analysis = analysis_store[analysis_id]

    return APIResponse(
        success=True,
        data=analysis.dict(),
        message=f"Analysis {analysis_id} status retrieved",
    )


@router.get("/results/{analysis_id}", response_model=APIResponse)
async def get_analysis_results(analysis_id: str):
    """Get the results of a completed analysis."""
    if analysis_id not in analysis_store:
        raise HTTPException(status_code=404, detail="Analysis not found")

    analysis = analysis_store[analysis_id]

    if analysis.status != AnalysisStatus.COMPLETED:
        raise HTTPException(
            status_code=400,
            detail=f"Analysis is not completed. Current status: {analysis.status}",
        )

    return APIResponse(
        success=True,
        data=analysis.results,
        message="Analysis results retrieved successfully",
    )


@router.delete("/cancel/{analysis_id}", response_model=APIResponse)
async def cancel_analysis(analysis_id: str):
    """Cancel a running analysis."""
    if analysis_id not in analysis_store:
        raise HTTPException(status_code=404, detail="Analysis not found")

    analysis = analysis_store[analysis_id]

    if analysis.status in [
        AnalysisStatus.COMPLETED,
        AnalysisStatus.FAILED,
        AnalysisStatus.CANCELLED,
    ]:
        raise HTTPException(
            status_code=400,
            detail=f"Cannot cancel analysis with status: {analysis.status}",
        )

    # Update status
    analysis.status = AnalysisStatus.CANCELLED
    analysis.completed_at = datetime.now()

    # Notify via WebSocket
    await websocket_manager.send_analysis_update(
        analysis_id, {"status": "cancelled", "message": "Analysis cancelled by user"}
    )

    return APIResponse(
        success=True,
        data={"analysis_id": analysis_id, "status": "cancelled"},
        message="Analysis cancelled successfully",
    )


@router.get("/history", response_model=APIResponse)
async def get_analysis_history(
    limit: int = 20,
    offset: int = 0,
    status: Optional[str] = None,
    current_user: Optional[Dict] = Depends(get_current_user),
):
    """Get analysis history with pagination and filtering."""
    try:
        # Filter analyses
        filtered_analyses = []
        for analysis in analysis_store.values():
            if status and analysis.status.value != status:
                continue
            filtered_analyses.append(analysis)

        # Sort by creation date (newest first)
        filtered_analyses.sort(key=lambda x: x.created_at, reverse=True)

        # Apply pagination
        total_count = len(filtered_analyses)
        paginated_analyses = filtered_analyses[offset : offset + limit]

        return APIResponse(
            success=True,
            data={
                "analyses": [analysis.dict() for analysis in paginated_analyses],
                "total_count": total_count,
                "limit": limit,
                "offset": offset,
                "has_more": offset + limit < total_count,
            },
            message="Analysis history retrieved successfully",
        )

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to retrieve history: {str(e)}"
        )


@router.get("/stream/{analysis_id}")
async def stream_analysis_progress(analysis_id: str):
    """Stream analysis progress via Server-Sent Events."""
    if analysis_id not in analysis_store:
        raise HTTPException(status_code=404, detail="Analysis not found")

    async def event_stream():
        """Generate Server-Sent Events for analysis progress."""
        while True:
            if analysis_id not in analysis_store:
                break

            analysis = analysis_store[analysis_id]

            # Send current status
            event_data = {
                "analysis_id": analysis_id,
                "status": analysis.status.value,
                "progress": analysis.progress,
                "timestamp": datetime.now().isoformat(),
            }

            yield f"data: {event_data}\n\n"

            # Stop streaming if analysis is complete
            if analysis.status in [
                AnalysisStatus.COMPLETED,
                AnalysisStatus.FAILED,
                AnalysisStatus.CANCELLED,
            ]:
                break

            # Wait before next update
            await asyncio.sleep(1)

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        },
    )


@router.post("/batch", response_model=APIResponse)
async def batch_analysis(
    requests: list[AnalysisRequest],
    background_tasks: BackgroundTasks,
    current_user: Optional[Dict] = Depends(get_current_user),
):
    """
    Start multiple analyses in batch.

    Useful for analyzing multiple portfolios or different configurations simultaneously.
    """
    try:
        if len(requests) > 10:  # Limit batch size
            raise HTTPException(status_code=400, detail="Batch size cannot exceed 10")

        analysis_ids = []

        for request in requests:
            # Generate unique analysis ID
            analysis_id = str(uuid.uuid4())

            # Create analysis record
            analysis_record = AnalysisResult(
                analysis_id=analysis_id,
                status=AnalysisStatus.PENDING,
                symbols=request.symbols,
                created_at=datetime.now(),
            )

            analysis_store[analysis_id] = analysis_record
            analysis_ids.append(analysis_id)

            # Start background task
            background_tasks.add_task(
                run_analysis_task, analysis_id, request, websocket_manager
            )

        return APIResponse(
            success=True,
            data={
                "analysis_ids": analysis_ids,
                "batch_size": len(requests),
                "status": "pending",
                "estimated_completion": f"{len(requests) * 3}-{len(requests) * 6} minutes",
            },
            message=f"Batch analysis of {len(requests)} requests started successfully",
        )

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to start batch analysis: {str(e)}"
        )


@router.get("/compare/{analysis_id1}/{analysis_id2}", response_model=APIResponse)
async def compare_analyses(analysis_id1: str, analysis_id2: str):
    """Compare results from two different analyses."""
    # Check both analyses exist and are completed
    for analysis_id in [analysis_id1, analysis_id2]:
        if analysis_id not in analysis_store:
            raise HTTPException(
                status_code=404, detail=f"Analysis {analysis_id} not found"
            )

        analysis = analysis_store[analysis_id]
        if analysis.status != AnalysisStatus.COMPLETED:
            raise HTTPException(
                status_code=400, detail=f"Analysis {analysis_id} is not completed"
            )

    try:
        analysis1 = analysis_store[analysis_id1]
        analysis2 = analysis_store[analysis_id2]

        # Basic comparison logic
        comparison = {
            "analysis_1": {
                "id": analysis_id1,
                "symbols": analysis1.symbols,
                "created_at": analysis1.created_at,
                "results_summary": "extracted_from_results",
            },
            "analysis_2": {
                "id": analysis_id2,
                "symbols": analysis2.symbols,
                "created_at": analysis2.created_at,
                "results_summary": "extracted_from_results",
            },
            "comparison": {
                "common_symbols": list(set(analysis1.symbols) & set(analysis2.symbols)),
                "unique_to_analysis_1": list(
                    set(analysis1.symbols) - set(analysis2.symbols)
                ),
                "unique_to_analysis_2": list(
                    set(analysis2.symbols) - set(analysis1.symbols)
                ),
                "time_difference": abs(
                    (analysis2.created_at - analysis1.created_at).total_seconds()
                ),
            },
        }

        return APIResponse(
            success=True, data=comparison, message="Analysis comparison completed"
        )

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to compare analyses: {str(e)}"
        )


@router.get("/metrics", response_model=APIResponse)
async def get_analysis_metrics():
    """Get analysis system metrics."""
    try:
        total_analyses = len(analysis_store)
        completed_analyses = sum(
            1 for a in analysis_store.values() if a.status == AnalysisStatus.COMPLETED
        )
        failed_analyses = sum(
            1 for a in analysis_store.values() if a.status == AnalysisStatus.FAILED
        )
        in_progress_analyses = sum(
            1 for a in analysis_store.values() if a.status == AnalysisStatus.IN_PROGRESS
        )

        metrics = {
            "total_analyses": total_analyses,
            "completed_analyses": completed_analyses,
            "failed_analyses": failed_analyses,
            "in_progress_analyses": in_progress_analyses,
            "success_rate": (
                completed_analyses / total_analyses if total_analyses > 0 else 0
            ),
            "websocket_stats": websocket_manager.get_stats(),
            "system_status": "operational",
        }

        return APIResponse(
            success=True,
            data=metrics,
            message="Analysis metrics retrieved successfully",
        )

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to retrieve metrics: {str(e)}"
        )
