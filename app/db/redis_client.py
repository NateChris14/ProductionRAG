"""
Shared async redis client with JSON helpers.
Used for: L2 Cache, rate limits, Celery broker/backend.
"""

import json
from typing import Any, Optional
import redis.asyncio as aioredis
from app.config import get_settings

_settings = get_settings()
_pool = aioredis.ConnectionPool.from_url(
    _settings.redis_url,
    max_connections=50,
    decode_responses=True,
)

def get_redis() -> aioredis.Redis:
    return aioredis.Redis(connection_pool=_pool)

async def cache_get(key: str) -> Optional[Any]:
    r = get_redis()
    raw = await r.get(key)
    if raw is None:
        return None
    try:
        return json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return raw

async def cache_set(key: str, value: Any, ttl: int = None) -> None:
    r = get_redis()
    ttl = ttl or _settings.cache_ttl_seconds
    await r.set(key, json.dumps(value), ex=ttl)

async def cache_delete(key: str) -> None:
    r = get_redis()
    await r.delete(key)






