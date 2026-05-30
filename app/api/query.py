"""
Query endpoints with SSE streaming and optional debug trace.

POST /query/stream - Server-Sent Events streaming answer
POST /query - Full JSON response (non-streaming, for batch/eval)
"""

from __future__ import annotations
import asyncio
import time
import json
import hashlib
from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from app.models.schemas import QueryRequest, QueryResponse
from app.services.query_processor import QueryProcessor
from app.services.retriever import HybridRetriever
from app.services.reranker import get_reranker
from app.services.context_builder import ContextBuilder
from app.services.llm_router import LLMRouter
from app.db.redis_client import cache_get, cache_set
from app.config import get_settings
import structlog

log = structlog.get_logger()
router = APIRouter(prefix="/query", tags=["query"])

_settings = get_settings()
_processor = QueryProcessor()
_retriever = HybridRetriever()
_reranker = get_reranker()
_ctx_builder = ContextBuilder(max_tokens=_settings.max_context_tokens)
_llm = LLMRouter()

def _cache_key(tenant_id: str, query: str, filters: dict) -> str:
    raw = f"{tenant_id}:{query}:{json.dumps(filters, sort_keys=True)}"
    return "qcache:" + hashlib.sha256(raw.encode()).hexdigest()[:20] 

async def _run_pipeline(req: QueryRequest) -> tuple:
    """
    Runs the full retrieval pipeline.
    Returns (context, rewrites, reranked_chunks, latency_map).
    """

    t = {}

    # Query rewrite
    t0 = time.perf_counter()
    rewrites = await _processor.rewrite(req.query)
    t["rewrite_ms"] = round((time.perf_counter() - t0) * 1000, 1)

    # Parallel retrieval across all query variants
    t0 = time.perf_counter()
    all_retrieval = await asyncio.gather(*[
        _retriever.retrieve(q, req.tenant_id, filters=req.filters) for q in rewrites
    ])

    # Flatten + deduplicate by id, preserving best-ranked order
    seen, merged = set(), []
    for hits in all_retrieval:
        for h in hits:
            if h.id not in seen:
                seen.add(h.id)
                merged.append(h)

    t["retrieval_ms"] = round((time.perf_counter() - t0) * 1000, 1)

    # Rerank
    t0 = time.perf_counter()
    reranked = await _reranker.rerank(
        req.query,
        merged[:_settings.rerank_candidates],
        top_n=_settings.rerank_top_n,
    )

    t["rerank_ms"] = round((time.perf_counter() - t0) * 1000, 1)

    # Compress context
    context = _ctx_builder.build(reranked)

    return context, rewrites, reranked, t

@router.post("/stream")
async def stream_query(req: QueryRequest):
    """SSE streaming endpoint - tokens arrive as they are generated."""

    cache_key = _cache_key(req.tenant_id, req.query, req.filters)
    cached = await cache_get(cache_key)

    async def event_stream():
        if cached:
            yield f"data: {json.dumps({'type': 'cache_hit', 'text': cached})}\n\n"
            yield "data: [DONE]\n\n"
            return

        context, rewrites, chunks, latency = await _run_pipeline(req)

        if req.debug:
            yield f"data: {json.dumps({'type': 'debug', 'rewrites': rewrites, 'latency': latency})}\n\n"

        full_answer = ""
        async for token in _llm.stream(req.query, context, req.chat_history):
            full_answer += token
            yield f"data: {json.dumps({'type': 'token', 'text': token})}\n\n"

        # Cache the completed answer
        await cache_set(cache_key, full_answer, ttl=_settings.cache_ttl_seconds)

        sources = [{"id": c.id, "score": c.score} for c in chunks[:5]]
        yield f"data: {json.dumps({'type': 'done', 'sources': sources})}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")

@router.post("", response_model=QueryResponse)
async def query(req: QueryRequest):
    """Non-streaming path for batch processing and eval harness."""

    cache_key = _cache_key(req.tenant_id, req.query, req.filters)
    cached = await cache_get(cache_key)

    if cached:
        return QueryResponse(
            answer=cached,
            chunks=[],
            model_used="cache",
            cache_hit=True
        )

    context, rewrites, chunks, latency = await _run_pipeline(req)
    answer = await _llm.complete(req.query, context, req.chat_history)
    await cache_set(cache_key, answer, ttl=_settings.cache_ttl_seconds)

    return QueryResponse(
        answer=answer,
        chunks=chunks,
        model_used=_llm.route(context),
        cache_hit=False,
        query_rewrites=rewrites,
        latency_ms=latency
    )