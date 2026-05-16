"""
Async batch embedder with content-hash cache and model versioning.

Key design decisions:
- Never embed synchronously inside the query path
- Deduplicate by content hash before calling the provider
- Cache embeddings in Redis keyed by (model_version, content_hash)
- Use provider batch limits: OpenAI text-embedding-3 supports 2048 inputs
"""

from __future__ import annotations
import asyncio
import hashlib
from typing import Optional
import openai
from tenacity import retry, stop_after_attempt, wait_exponential
from app.config import get_settings
from app.db.redis_client import cache_get, cache_set
import structlog

log = structlog.get_logger()
_settings = get_settings()

def _content_hash(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:16]

def _embed_cache_key(model: str, content_hash: str) -> str:
    return f"emb:{model}:{content_hash}"

class AsyncBatchEmbedder:
    """
    Batches texts, checks cache, calls the embedding provider for misses,
    and updates the cache. Fully async.

    Usage:
        embedder = AsyncBatchEmbedder()
        vectors = await embedder.embed(["text 1","text 2", ...])
    """

    def __init__(self, model: Optional[str] = None, batch_size: Optional[int] = None, cache_ttl: int = 86400 * 7,) -> None:

        self.model = model or _settings.embed_model
        self.batch_size = batch_size or _settings.embed_batch_size
        self.cache_ttl = cache_ttl
        self._client = openai.AsyncOpenAI(api_key=_settings.openai_api_key)

    async def embed(self, texts: list[str]) -> list[list[float]]:
        """Returns vectors in the same order as input texts."""

        if not texts:
            return []

        hashes = [_content_hash(t) for t in texts]
        keys = [_embed_cache_key(self.model, h) for h in hashes]

        # Check cache for all texts in parallel
        cached_results = await asyncio.gather(*[cache_get(k) for k in keys])

        misses_idx = [i for i, v in enumerate(cached_results) if v is None]
        miss_texts = [texts[i] for i in misses_idx]

        if miss_texts:
            miss_vectors = await self._embed_batched(miss_texts)
            # Write back to cache
            await asyncio.gather(*[
                cache_set(keys[misses_idx[i]], miss_vectors[i], ttl=self.cache_ttl)
                for i in range(len(miss_texts))
            ])

            for i, vec in zip(misses_idx, miss_vectors):
                cached_results[i] = vec

        return cached_results # type: ignore[return-value]

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10))
    async def _embed_batched(self, texts: list[str]) -> list[list[float]]:
        """Calls OpenAI in provider-sized batches with exponential backoff retry."""

        all_vectors: list[list[float]] = []

        for start in range(0, len(texts), self.batch_size):
            batch = texts[start: start + self.batch_size]
            log.info("embedding_batch", size=len(batch), model=self.model)
            response = await self._client.embeddings.create(
                model=self.model,
                input=batch,
            )

            all_vectors.extend([item.embedding for item in response.data])

        return all_vectors

    async def embed_query(self, query: str) -> list[float]:
        """Single query path with cache. Used in the serving pipeline."""

        vecs = await self.embed([query])
        return vecs[0]
