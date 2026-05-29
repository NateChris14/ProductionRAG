"""
Orchestrates the full ingestion pipeline for a single document.
Called by celery workers - never in the request path.

Steps:

1. Parse + normalize content
2. Semantic chunk
3. Batch embed (cached)
4. Upsert to vector store
5. Update Postgres job status
"""

from __future__ import annotations
import uuid
import hashlib
from datetime import datetime, timezone
from app.pipeline.chunker import SemanticChunker
from app.pipeline.embedder import AsyncBatchEmbedder
from app.services.vector_store import get_vector_store
from app.models.schemas import Chunk, ChunkMetadata, DocumentIngestionRequest
from app.db.postgres import AsyncSessionFactory
from app.models.orm import Document, DocumentChunk, IngestionJob
import structlog

log = structlog.get_logger()

_chunker = SemanticChunker(min_tokens=150, max_tokens=700)
_embedder = AsyncBatchEmbedder()
_vector_store = get_vector_store()

def _content_hash(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()

async def run_ingestion(req: DocumentIngestionRequest, job_id: str) -> None:
    doc_id = str(uuid.uuid4())

    # Persist document row + mark job running
    async with AsyncSessionFactory() as session:
        doc = Document(
            id=doc_id,
            tenant_id=req.tenant_id,
            title=req.title,
            source_type=req.source_type,
            source_uri=req.source_uri,
            content_hash=_content_hash(req.content),
            acl_tags=req.acl_tags,
            extra_meta=req.metadata
        )
        session.add(doc)

        job = await session.get(IngestionJob, job_id)
        if job:
            job.status = "running"
            job.stages = {"started_at": datetime.now(timezone.utc).isoformat()}
        await session.commit()

    # Chunk
    raw_chunks = _chunker.chunk(req.content, document_id=doc_id)
    log.info("chunked", doc_id=doc_id, count=len(raw_chunks))

    # Embed (batch + cached)
    texts = [c.text for c in raw_chunks]
    vectors = await _embedder.embed(texts)

    # Build Chunk Objects
    now = datetime.now(timezone.utc).isoformat()
    chunks: list[Chunk] = []
    for raw, vec in zip(raw_chunks, vectors):
        meta = ChunkMetadata(
            tenant_id=req.tenant_id,
            document_id=doc_id,
            chunk_index=raw.chunk_index,
            parent_id=raw.parent_id,
            source_type=req.source_type,
            source_uri=req.source_uri,
            title=req.title,
            token_count=raw.token_count,
            acl_tags=req.acl_tags,
            created_at=now,
        )

        chunks.append(Chunk(id=raw.id, text=raw.text, metadata=meta, embedding=vec))

    # Write to vector store
    await _vector_store.upsert(chunks)
    log.info("vectors_upserted", doc_id=doc_id, count=len(chunks))

    # Persist chunk metadata to Postgres + mark job done
    async with AsyncSessionFactory() as session:
        for c in chunks:
            session.add(DocumentChunk(
                id=c.id,
                document_id=doc_id,
                tenant_id=req.tenant_id,
                chunk_index=c.metadata.parent_id,
                text=c.text,
                token_count=c.metadata.token_count,
                vector_id=c.id,
                embed_model=_embedder.model,
                acl_tags=req.acl_tags,
            ))

        job = await session.get(IngestionJob, job_id)
        if job:
            job.status = "done"
            job.stages = {
                **(job.stages or {}),
                "completed_at": datetime.now(timezone.utc).isoformat(),
                "chunk_count": len(chunks),
            }
        await session.commit()

    log.info("ingestion_complete", doc_id=doc_id, chunks=len(chunks))

    