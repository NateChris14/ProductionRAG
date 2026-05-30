"""
FastAPI application entry point.
"""

from contextlib import asynccontextmanager
from fastapi import FastAPI
from app.api import ingest, query, health
from app.db.postgres import init_db
from app.utils.logging import configure_logging
from app.utils.tracing import configure_tracing
from app.config import get_settings

_settings = get_settings()

@asynccontextmanager
async def lifespan(app: FastAPI):
    configure_logging(_settings.log_level)
    await init_db()
    configure_tracing(app)
    yield

app = FastAPI(
    title="Production RAG API",
    version="1.0.0",
    lifespan=lifespan,
)

app.include_router(health.router)
app.include_router(ingest.router)
app.include_router(query.router)