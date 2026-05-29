"""
Ingestion endpoints.

POST /ingest - enqueue a document for async processing
GET /ingest/jobs/{id} - poll ingestion job status
"""

from __future__ import annotations
import uuid
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from app.models.schemas import DocumentIngestionRequest, IngestResponse
from app.models.orm import IngestionJob
from app.db.postgres import get_db
from sqlalchemy.ext.asyncio import AsyncSession
import structlog

from worker.tasks import ingest_document

log = structlog.get_logger()
router = APIRouter(prefix="/ingest", tags=["ingestion"])

@router.post("", response_model=IngestResponse, status_code=202)
async def ingest(req: DocumentIngestionRequest, db: AsyncSession = Depends(get_db)):

    job_id = str(uuid.uuid4())
    doc_id = str(uuid.uuid4())

    # Persist job record immediately so GET /jobs/{id} works right away
    job = IngestionJob(
        id=job_id,
        document_id=doc_id,
        tenant_id=req.tenant_id,
        status="queued",
    )

    db.add(job)
    await db.commit()

    payload = req.model_dump()
    payload["job_id"] = job_id

    ingest_document.apply_async(
        args=[payload],
        queue="ingestion",
        task_id=job_id,
    )

    log.info("ingestion_queued", job_id=job_id, tenant_id=req.tenant_id)
    return IngestResponse(job_id=job_id, document_id=doc_id)

@router.get("/jobs/{job_id}")
async def get_job(job_id: str, db: AsyncSession = Depends(get_db)):
    
    result = await db.execute(
        select(IngestionJob).where(IngestionJob.id == job_id)
    )

    job = result.scalar_one_or_none()
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return {
        "job_id": job.id,
        "status": job.status,
        "stages": job.stages,
        "error": job.error,
    }



    