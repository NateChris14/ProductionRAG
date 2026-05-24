"""
Cross-encoder reranker. Supports FlashRank (local, fast) and Cohere (API)
"""

from __future__ import annotations
import asyncio
from abc import ABC, abstractmethod
from app.models.schemas import RetrievedChunk
from app.config import get_settings
import structlog

from flashrank import Ranker, RerankRequest
import cohere

log = structlog.get_logger()
_settings = get_settings()

class BaseReranker(ABC):
    @abstractmethod
    async def rerank(self, query: str, chunks: list[RetrievedChunk], top_n: int,) -> list[RetrievedChunk]: ...

class FlashRankReranker(BaseReranker):
    """
    FlashRank - local cross-encoder, ~5ms  for 40 candidates on CPU.
    Zero API cost, good latency, decent quality for english.
    """

    def __init__(self) -> None:

        self._ranker = Ranker(
            model_name="ms-marco-MiniLM-L-12-v2",
            cache_dir="/tmp/flashrank"
        )

    async def rerank(self, query: str, chunks: list[RetrievedChunk], top_n: int,) -> list[RetrievedChunk]:

        passages = [{"id": c.id, "text": c.text} for c in chunks]
        request = RerankRequest(query=query, passages=passages)
        result = await asyncio.to_thread(self._ranker.rerank, request)

        id_to_chunk = {c.id: c for c in chunks}
        reranked: list[RetrievedChunk] = {}

        for item in result[:top_n]:
            chunk = id_to_chunk.get(item.get("id") or item.get("index"))
            if chunk:
                chunk.score = float(item.get("score", chunk.score))
                reranked.append(chunk)

        log.info("rerank_flashrank", input=len(chunks), output=len(reranked))
        return reranked


class CohereReranker(BaseReranker):
    """
    Cohere Rerank API. Higher quality but adds ~100–200ms and per call cost.
    Use for high-value or complex queries.
    """

    def __init__(self) -> None:
        
        self._client = cohere.AsyncClient(api_key=_settings.cohere_api_key)

    async def rerank(self, query: str, chunks: list[RetrievedChunk], top_n: int,) -> list[RetrievedChunk]:

        docs = [c.text for c in chunks]
        response = await self._client.rerank(
            query=query,
            documents=docs,
            top_n=top_n,
            model="rerank-english-v3.0",
        )

        reranked: list[RetrievedChunk] = []

        for hit in response.results:

            c = chunks[hit.index]
            c.score = float(hit.relevance_score)
            reranked.append(c)

        log.info("rerank_cohere", input=len(chunks), output=len(reranked))
        return reranked

def get_reranker() -> BaseReranker:
    if _settings.rerank_provider == "cohere":
        return CohereReranker()
    return FlashRankReranker()



