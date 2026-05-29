"""
Health check endpoint for load balancer probes and uptime monitoring.
"""

from fastapi import APIRouter
from app.models.schemas import HealthResponse
from app.db.redis_client import get_redis
import structlog

log = structlog.get_logger()
router = APIRouter(tags=["health"])

@router.get("/health", response_model=HealthResponse)
async def health():

    checks: dict[str, str] = {}

    try:
        await get_redis().ping()
        checks["redis"] = "ok"
    except Exception as e:
        checks["redis"] = f"error: {e}"

    status = "ok" if all(v == "ok" for v in checks.values()) else "degraded"
    return HealthResponse(status=status, checks=checks)