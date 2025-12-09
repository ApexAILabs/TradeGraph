import asyncio
import time
import psutil
from datetime import datetime
from typing import Dict

from fastapi import APIRouter, HTTPException

from ..models import APIResponse, HealthCheckResponse, SystemMetrics

router = APIRouter()

# Store startup time for uptime calculation
startup_time = time.time()


@router.get("/", response_model=APIResponse)
async def health_check():
    """Basic health check endpoint."""
    return APIResponse(
        success=True,
        data={
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "uptime_seconds": time.time() - startup_time,
        },
        message="Service is healthy",
    )


@router.get("/detailed", response_model=APIResponse)
async def detailed_health_check():
    """Detailed health check with system metrics."""
    try:
        # Calculate uptime
        uptime_seconds = time.time() - startup_time

        # Get system metrics
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage("/")

        # Test critical services
        services_status = await test_services()

        # Overall health determination
        is_healthy = all(services_status.values()) and cpu_percent < 90 and memory.percent < 90

        health_data = HealthCheckResponse(
            status="healthy" if is_healthy else "unhealthy",
            version="1.0.0",
            uptime=uptime_seconds,
            services=services_status,
            system_info={
                "cpu_usage_percent": cpu_percent,
                "memory": {
                    "total_gb": round(memory.total / (1024**3), 2),
                    "available_gb": round(memory.available / (1024**3), 2),
                    "used_percent": memory.percent,
                },
                "disk": {
                    "total_gb": round(disk.total / (1024**3), 2),
                    "free_gb": round(disk.free / (1024**3), 2),
                    "used_percent": round((disk.used / disk.total) * 100, 1),
                },
            },
        )

        return APIResponse(
            success=True, data=health_data.dict(), message="Detailed health check completed"
        )

    except Exception as e:
        return APIResponse(
            success=False, data=None, message="Health check failed", error={"detail": str(e)}
        )


async def test_services() -> Dict[str, str]:
    """Test availability of critical services."""
    services = {}

    # Test database connection (simulated)
    try:
        # In a real app, this would test actual database connectivity
        await asyncio.sleep(0.1)  # Simulate DB query
        services["database"] = "healthy"
    except Exception:
        services["database"] = "unhealthy"

    # Test external APIs
    try:
        # Test if we can import and initialize core components
        from tradegraph_financial_advisor import FinancialAdvisor

        FinancialAdvisor()
        services["financial_advisor"] = "healthy"
    except Exception:
        services["financial_advisor"] = "unhealthy"

    # Test OpenAI connectivity (simulated)
    try:
        import os

        if os.getenv("OPENAI_API_KEY"):
            services["openai_api"] = "configured"
        else:
            services["openai_api"] = "not_configured"
    except Exception:
        services["openai_api"] = "error"

    return services


@router.get("/metrics", response_model=APIResponse)
async def get_system_metrics():
    """Get detailed system performance metrics."""
    try:
        # CPU metrics
        cpu_percent = psutil.cpu_percent(interval=1)
        cpu_count = psutil.cpu_count()

        # Memory metrics
        memory = psutil.virtual_memory()

        # Disk metrics
        disk = psutil.disk_usage("/")

        # Network metrics (if available)
        try:
            network = psutil.net_io_counters()
            network_stats = {
                "bytes_sent": network.bytes_sent,
                "bytes_recv": network.bytes_recv,
                "packets_sent": network.packets_sent,
                "packets_recv": network.packets_recv,
            }
        except Exception:
            network_stats = {}

        # Process metrics
        process = psutil.Process()
        process_memory = process.memory_info()

        metrics = SystemMetrics(
            cpu_usage=cpu_percent,
            memory_usage=memory.percent,
            active_connections=0,  # Would be tracked by WebSocket manager
            total_requests=0,  # Would be tracked by middleware
            average_response_time=0.0,  # Would be tracked by middleware
            error_rate=0.0,  # Would be tracked by middleware
            uptime_seconds=time.time() - startup_time,
        )

        detailed_metrics = {
            "system_metrics": metrics.dict(),
            "detailed_stats": {
                "cpu": {
                    "usage_percent": cpu_percent,
                    "core_count": cpu_count,
                    "load_average": (
                        list(psutil.getloadavg()) if hasattr(psutil, "getloadavg") else []
                    ),
                },
                "memory": {
                    "total_gb": round(memory.total / (1024**3), 2),
                    "available_gb": round(memory.available / (1024**3), 2),
                    "used_gb": round(memory.used / (1024**3), 2),
                    "used_percent": memory.percent,
                    "cached_gb": round(getattr(memory, "cached", 0) / (1024**3), 2),
                },
                "disk": {
                    "total_gb": round(disk.total / (1024**3), 2),
                    "free_gb": round(disk.free / (1024**3), 2),
                    "used_gb": round(disk.used / (1024**3), 2),
                    "used_percent": round((disk.used / disk.total) * 100, 1),
                },
                "process": {
                    "memory_rss_mb": round(process_memory.rss / (1024**2), 2),
                    "memory_vms_mb": round(process_memory.vms / (1024**2), 2),
                    "cpu_percent": process.cpu_percent(),
                    "num_threads": process.num_threads(),
                    "create_time": datetime.fromtimestamp(process.create_time()).isoformat(),
                },
                "network": network_stats,
            },
        }

        return APIResponse(
            success=True, data=detailed_metrics, message="System metrics retrieved successfully"
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to retrieve metrics: {str(e)}")


@router.get("/readiness", response_model=APIResponse)
async def readiness_check():
    """Kubernetes-style readiness probe."""
    try:
        # Check if the service is ready to handle requests
        services_status = await test_services()

        # Service is ready if core components are healthy
        core_services = ["financial_advisor"]
        is_ready = all(services_status.get(service) == "healthy" for service in core_services)

        if is_ready:
            return APIResponse(
                success=True,
                data={"status": "ready", "services": services_status},
                message="Service is ready",
            )
        else:
            return APIResponse(
                success=False,
                data={"status": "not_ready", "services": services_status},
                message="Service is not ready",
            )

    except Exception as e:
        return APIResponse(
            success=False, data=None, message="Readiness check failed", error={"detail": str(e)}
        )


@router.get("/liveness", response_model=APIResponse)
async def liveness_check():
    """Kubernetes-style liveness probe."""
    try:
        # Simple check to verify the service is alive
        current_time = datetime.now()
        uptime_seconds = time.time() - startup_time

        # Service is alive if it's been running and can respond
        is_alive = uptime_seconds > 0

        return APIResponse(
            success=is_alive,
            data={
                "status": "alive" if is_alive else "dead",
                "uptime_seconds": uptime_seconds,
                "current_time": current_time.isoformat(),
            },
            message="Liveness check completed",
        )

    except Exception as e:
        return APIResponse(
            success=False, data=None, message="Liveness check failed", error={"detail": str(e)}
        )


@router.get("/dependencies", response_model=APIResponse)
async def check_dependencies():
    """Check status of external dependencies."""
    try:
        dependencies = {}

        # Check Python packages
        try:
            import pkg_resources

            installed_packages = {pkg.key: pkg.version for pkg in pkg_resources.working_set}

            required_packages = [
                "fastapi",
                "uvicorn",
                "pydantic",
                "langchain",
                "langgraph",
                "pandas",
                "numpy",
                "aiohttp",
            ]

            for package in required_packages:
                if package in installed_packages:
                    dependencies[f"python_{package}"] = {
                        "status": "installed",
                        "version": installed_packages[package],
                    }
                else:
                    dependencies[f"python_{package}"] = {"status": "missing", "version": None}

        except Exception as e:
            dependencies["python_packages"] = {"status": "error", "error": str(e)}

        # Check environment variables
        import os

        env_vars = ["OPENAI_API_KEY", "FINNHUB_API_KEY"]
        for var in env_vars:
            dependencies[f"env_{var.lower()}"] = {
                "status": "configured" if os.getenv(var) else "missing"
            }

        # Check external services (simplified)
        dependencies["external_apis"] = {
            "openai": "configured" if os.getenv("OPENAI_API_KEY") else "not_configured",
            "finnhub": "configured" if os.getenv("FINNHUB_API_KEY") else "not_configured",
        }

        return APIResponse(success=True, data=dependencies, message="Dependencies check completed")

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to check dependencies: {str(e)}")


@router.post("/maintenance", response_model=APIResponse)
async def toggle_maintenance_mode():
    """Toggle maintenance mode (placeholder for production use)."""
    # In production, this would toggle a maintenance flag
    # and potentially drain connections gracefully

    return APIResponse(
        success=True,
        data={"maintenance_mode": False, "message": "Maintenance mode not implemented"},
        message="Maintenance mode toggle completed",
    )
