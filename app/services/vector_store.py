"""
Distributed vector store abstraction for Milvus and Pinecone.

Tenant isolation: each tenant gets its own partition (Milvus) or namespace (Pinecone).
All metadata filters are enforced at query time, not post-hoc,
to prevent cross-tenant data leakage.
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Optional, Any
from app.config import get_settings
from app.models.schemas import Chunk, RetrievedChunk
import structlog
import asyncio

from pymilvus import MilvusClient, DataType

log = structlog.get_logger()
_settings = get_settings()

COLLECTION_NAME = "rag_chunks"

class BaseVectorStore(ABC):
    @abstractmethod
    async def upsert(self, chunks: list[Chunk]) -> None: ...

    @abstractmethod
    async def search(
        self,
        vector: list[float],
        top_k: int,
        tenant_id: str,
        filters: dict[str, Any],

    ) -> list[RetrievedChunk]: ...

    @abstractmethod
    async def delete_by_document(self, document_id: str, tenant_id: str) -> None: ...

class MilvusVectorStore(BaseVectorStore):
    """
    Milvus 2.5 with:
    - Partitions per tenant for query-time isolation
    - HNSW index for dense ANN
    - Native BM25 full-text field (Milvus 2.5+) for sparse mode
    - JSON metadata payload for filter expressions
    """

    def __init__(self) -> None:

        self._client = MilvusClient(
            uri=_settings.milvus_uri,
            token=_settings.milvus_token or None,
        )

        self._ensure_collection()

    def _ensure_collection(self) -> None:

        if self._client.has_collection(COLLECTION_NAME):
            return
        
        schema = self._client.create_schema(auto_id=False,
        enable_dynamic_field=True)
        schema.add_field("id", DataType.VARCHAR, max_length=64, is_primary=True)
        schema.add_field("tenant_id", DataType.VARCHAR, max_length=128)
        schema.add_field("document_id", DataType.VARCHAR, max_length=64)
        schema.add_field("text", DataType.VARCHAR, max_length=65535, enable_analyzer=True) # BM25 full-text (Milvus 2.5+)
        schema.add_field("vector", DataType.FLOAT_VECTOR, dim=_settings.embed_dims)
        schema.add_field("acl_tags", DataType.JSON)
        schema.add_field("meta", DataType.JSON)

        index_params = self._client.prepare_index_params()
        index_params.add_index(
            field_name="vector",
            index_type="HNSW",
            metric_type="COSINE",
            params={"M": 16, "efConstruction": 256},
        )

        index_params.add_index(
            field_name="text",
            index_type="INVERTED", # BM25 sparse index
        )

        self._client.create_collection(
            collection_name=COLLECTION_NAME,
            schema=schema,
            index_params=index_params,
        )

        log.info("milvus_collection_created", name=COLLECTION_NAME)

    async def upsert(self, chunks: list[Chunk]) -> None:

        data = [
            {
                "id": c.id,
                "tenant_id": c.metadata.tenant_id,
                "document_id": c.metadata.document_id,
                "text": c.text,
                "vector": c.embedding,
                "acl_tags": c.metadata.acl_tags,
                "meta": {
                    "title": c.metadata.title,
                    "source_uri": c.metadata.source_uri,
                    "chunk_index": c.metadata.chunk_index,
                    "token_count": c.metadata.token_count,
                },
            }
            for c in chunks
            if c.embedding is not None
        ]

        await asyncio.to_thread(
            self._client.upsert,
            collection_name=COLLECTION_NAME,
            data=data,
        )

    async def search(self, vector: list[float], top_k: int, tenant_id: str, filters: dict[str, Any]) -> lits[RetrievedChunk]:

        expr = f'tenant_id == "{tenant_id}"'

        if "source_type" in filters:
            expr += f' && meta["source_type"] == "{filters["source_type"]}"'

        raw = await asyncio.to_thread(
            self._client.search,
            collection_name=COLLECTION_NAME,
            data=[vector],
            anns_field="vector",
            limit=top_k,
            filter=expr,
            output_fields=["id", "text", "meta", "acl_tags"],
        )

        results = []
        for hit in raw:
            results.append(RetrievedChunk(
                id=hit["id"],
                text=hit["entity"]["text"],
                score=float(hit["distance"]),
                metadata=hit["entity"].get("meta", {}),
            ))

        return results

    async def delete_by_document(self, document_id: str, tenant_id: str) -> None:
        expr = f'document_id == "{document_id}" && tenant_id == "{tenant_id}"'
        await asyncio.to_thread(
            self._client.delete,
            collection_name=COLLECTION_NAME,
            filter=expr,
        )




    





