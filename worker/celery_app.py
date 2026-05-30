"""
Celery application configuration.

Separate queues for:
 - ingestion: document parse + chunk + embed (heavy, slow)
 - reindex: bulk re-embedding when model changes
 - eval: offline evaluation jobs

Workers are deployed as separate containers so they scale independently.
"""

from celery import Celery
from app.config import get_settings

_settings = get_settings()

celery_app = Celery(
    "production_rag",
    broker=_settings.redis_url,
    backend=_settings.redis_url,
)

celery_app.conf.update(
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    timezone="UTC",
    enable_utc=True,
    task_track_started=True,
    task_acks_late=True, # re-queue on worker crash
    worker_prefetch_multiplier=1, # fair dispatch for long tasks
    task_routes={
        "worker.tasks.ingest_document": {"queue": "ingestion"},
        "worker.tasks.reindex_tenant": {"queue": "reindex"},
        "worker.tasks.run_eval": {"queue": "eval"},
    },
)