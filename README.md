production-rag/
├── main.py
├── pyproject.toml
├── Dockerfile
├── docker-compose.yml
├── .env.example
├── app/
│   ├── config.py
│   ├── models/
│   │   ├── schemas.py         # Pydantic request/response models
│   │   └── orm.py             # SQLAlchemy ORM (Document, Chunk, Job)
│   ├── db/
│   │   ├── postgres.py        # Async engine + session factory
│   │   └── redis_client.py    # Async Redis + JSON cache helpers
│   ├── pipeline/
│   │   ├── chunker.py         # Semantic + dynamic chunker
│   │   └── embedder.py        # Async batch embedder + content-hash cache
│   ├── services/
│   │   ├── vector_store.py    # Milvus + Pinecone abstraction
│   │   ├── retriever.py       # Hybrid retrieval + RRF
│   │   ├── reranker.py        # FlashRank + Cohere rerankers
│   │   ├── query_processor.py # Normalize + LLM rewrite
│   │   ├── context_builder.py # Dedup + token-budget packing
│   │   ├── llm_router.py      # Model routing + async streaming
│   │   └── ingestion_service.py # Full ingest orchestration
│   ├── api/
│   │   ├── ingest.py          # POST /ingest, GET /jobs/{id}
│   │   ├── query.py           # POST /query/stream, POST /query
│   │   └── health.py          # GET /health
│   ├── utils/
│   │   ├── logging.py         # Structlog JSON config
│   │   └── tracing.py         # OTel + Langfuse setup
│   └── eval/
│       └── evaluator.py       # Recall@k, MRR, latency eval harness
├── worker/
│   ├── celery_app.py          # Celery config + queues
│   └── tasks.py               # ingest_document, reindex_tenant
└── tests/
    └── test_pipeline.py       # Chunker, RRF, context builder tests