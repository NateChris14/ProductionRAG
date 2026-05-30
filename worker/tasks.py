"""
Celery task definitions.

All async work is wrapped with asyncio.run() since Celery workers
run in a synchronous context.
"""

from __future__ import annotations
import asyncio
import uuid
from worker.celery_app import celery_app
from app.models.schemas import DocumentIngestionRequest
import structlog

from app.services.ingestion_service import run_ingestion
from app.pipeline.embedder import AsyncBatchEmbedder
from app.services.vector_store import get_vector_store
from app.db.postgres import AsyncSessionFactory
from app.models.orm import DocumentChunk
from app.models.schemas import Chunk, ChunkMetadata
from sqlalchemy import select

log = structlog.get_logger()

@celery_app.task(
    name="worker.tasks.ingest_document",
    bind=True,
    max_retries=3,
    default_retry_delay=30,
    queue="ingestion",
)
def ingest_document(self, payload: dict) -> dict:
    """
    Entry point for document ingestion.
    payload is a serialized DocumentIngestionRequest dict + job_id.
    """

    job_id = payload.pop("job_id", str(uuid.uuid4()))
    req = DocumentIngestionRequest(**payload)

    try:
        asyncio.run(run_ingestion(req, job_id))
        return {"status": "done", "job_id": job_id}
    except Exception as exc:
        log.error("ingestion_task_failed", job_id=job_id, error=str(exc))
        raise self.retry(exc=exc)

@celery_app.task(
    name="worker.tasks.reindex_tenant",
    bind=True,
    max_retries=2,
    queue="reindex",
)
def reindex_tenant(self, tenant_id: str, new_model: str) -> dict:
    """
    Re-embeds all active chunks for a tenant when the embedding model changes.
    Fetches chunks in batches from Postgres, re-embeds, and upserts.
    """

    async def _reindex():

        embedder = AsyncBatchEmbedder(model=new_model)
        vs = get_vector_store()
        batch_size = 256

        async with AsyncSessionFactory() as session:
            result = await session.execute(
                select(DocumentChunk)
                .where(DocumentChunk.tenant_id == tenant_id)
                .where(DocumentChunk.is_active == True)
            )
            all_chunks = result.scalars().all()

            for i in range(0, len(all_chunks), batch_size):

                batch = all_chunks[i: i + batch_size]
                texts = [c.text for c in batch]
                vectors = await embedder.embed(texts)

                chunk_objs = [
                    Chunk(
                        id=c.id,
                        text=c.text,
                        metadata=ChunkMetadata(
                            tenant_id=c.tenant_id,
                            document_id=c.document_id,
                            chunk_index=c.chunk_index,
                            token_count=c.token_count,
                            source_type="text",
                            acl_tags=c.acl_tags or [],
                        ),
                        embedding=vec,
                    )
                    for c, vec in zip(batch, vectors)
                ]
                await vs.upsert(chunk_objs)
                log.info("reindex_batch", tenant=tenant_id, done=i + len(batch))

    try:
        asyncio.run(_reindex())
        return {"status": "done", "tenant_id": tenant_id}
    except Exception as exc:
        log.error("reindex_failed", tenant_id=tenant_id, error=str(exc))
        raise self.retry(exc=exc)





