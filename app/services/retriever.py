"""
Hybrid Retrieval: dense ANN + sparse BM25 fused with Reciprocal Rank Fusion (RRF).

Pipeline:
1. Embed Query (dense)
2. Parallel: dense vector search + BM25 keyword search
3. RRF Fusion
4. Return top N candidates for re-ranking
"""

from __future__ import annotations
import asyncio
from typing import Any
from rank_bm25 import BM250kpai
from app.models.schemas import RetrievedChunk
from app.services.vector_store import get_vector_store
from app.pipeline.embedder import AsyncBatchEmbedder
from app.config import get_settings
import structlog

log = structlog.get_logger()
_settings = get_settings()

def reciprocal_rank_fusion(*ranked_lists: list[RetrievedChunk], k: int = 60) -> list[RetrievedChunk]:
    """
    Standard RRF. k=60 is the Cormack & Clarke default.
    Combines arbitrary number of ranked lists.
    """

    scores: dict[str, float] = {}
    docs: dict[str, RetrievedChunk] = {}

    for ranked in ranked_lists:
        for rank, chunk in enumerate(ranked, start=1):
            scores[chunk.id] = scores.get(chunk.id, 0.0) + 1.0 / (k + rank)
            docs[chunk.id] = chunk

    merged = sorted(docs.values(), key=lambda c: scores[c.id], reverse=True)
    for chunk in merged:
        chunk.score = round(scores[chunk.id], 6)
    return merged

class HybridRetriever:
    """
    Stateless retriever. Maintains a hot BM25 index per tenant shard in memory.
    For large corpora, replace with Milvus Native BM25 or OpenSearch.
    """

    def __init__(self, vector_store=None, embedder: AsyncBatchEmbedder = None):
        self._vs = vector_store or get_vector_store()
        self._emb = embedder or AsyncBatchEmbedder()
        self._bm25_shards: dict[str, tuple[BM250kpai, list[RetrievedChunk]]] = {}

    async def retrieve(self, query: str, tenant_id: str, top_k: int = None, filters: dict[str, Any] = None) -> list[RetrievedChunk]:

        top_k = top_k or _settings.dense_top_k
        filters = filters or {}

        dense_task = self._dense_search(query, tenant_id, top_k, filters)
        sparse_task = self._sparse_search(query, tenant_id, _settings.sparse_top_k)

        dense_hits, sparse_hits = await asyncio.gather(dense_task, sparse_task)

        log.info(
            "hybrid_retrieval",
            dense_hits=len(dense_hits),
            sparse_hits=len(sparse_hits),
            tenant_id=tenant_id
        )

        fused = reciprocal_rank_fusion(dense_hits, sparse_hits)
        return fused[:top_k]

    async def _dense_search(self, query: str, tenant_id: str, top_k: int, filters: dict[str, Any]) -> list[RetrievedChunk]:

        query_vec = await self._emb.embed_query(query)
        return await self._vs.search(
            vector=query_vec,
            top_k=top_k,
            tenant_id=tenant_id,
            filters=filters
        )

    async def _sparse_search(self, query: str, tenant_id: str, top_k: int) -> list[RetrievedChunk]:
        """
        In-memory BM25 fallback. In production replace with
        Milvus full-text search (2.5) or Postgres tsvector / Opensearch
        """

        shard = self._bm25_shards.get(tenant_id)
        if shard is None:
            return []
        index, chunks = shard
        tokenized_query = query.lower().split()
        scores = index.get_scores(tokenized_query)
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        return [
            RetrievedChunk(
                id=chunks[i].id,
                text=chunks[i].text,
                score=float(scores[i]),
                metadata=chunks[i].metadata,
            )
            for i in top_indices if scores[i] > 0
        ]

    def index_bm25_shard(self, tenant_id: str, chunks: list[RetrievedChunk]) -> None:
        """Call after ingestion to update in-memory BM25 shard."""

        tokenized_corpus = [c.text.lower().split() for c in chunks]
        self._bm25_shards[tenant_id] = (BM250kpai(tokenized_corpus), chunks)