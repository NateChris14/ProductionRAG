"""
Pydantic schemas for API request, responses and internal data transfer objects.
"""

from __future__ import annotations
from pydantic import BaseModel, Field, field_validator
from typing import Optional, Any
import uuid

# Ingestion

class DocumentIngestionRequest(BaseModel):
    tenant_id: str
    source_type: str = "text" # text / pdf / html / markdown
    content: str
    title: Optional[str] = None
    source_uri: Optional[str] = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    acl_tags: list[str] = Field(default_factory=list)

class IngestResponse(BaseModel):
    job_id: str
    document_id: str
    status: str = "queued"

# Chunks

class ChunkMetadata(BaseModel):
    tenant_id: str
    document_id: str
    chunk_index: int
    parent_id: Optional[str] = None
    source_type: str
    source_uri: Optional[str] = None
    title: Optional[str] = None
    token_count: int
    language: str = "en"
    acl_tags: list[str] = Field(default_factory=list)
    created_at: Optional[str] = None

class Chunk(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    text: str
    metadata: ChunkMetadata
    embedding: Optional[list[float]] = None

# Query / Serving

class QueryRequest(BaseModel):
    tenant_id: str
    user_id: str
    query: str = Field(..., min_length=1, max_length=4096)
    chat_history: list[dict[str, str]] = Field(default_factory=list)
    filters: dict[str, Any] = Field(default_factory=dict)
    stream: bool = True
    debug: bool = False

    @field_validator("chat_history")
    @classmethod
    def limit_history(cls, v: list) -> list:
        return v[-10:] # keep the last 10 turns

class RetrievedChunk(BaseModel):
    id: str
    text: str
    score: float
    metadata: dict[str, Any]

class QueryResponse(BaseModel):
    answer: str
    chunks: list[RetrievedChunk]
    model_used: str
    cache_hit: bool = False
    query_rewrites: list[str] = Field(default_factory=list)
    latency_ms: dict[str, float] = Field(default_factory=dict)

# Health

class HealthResponse(BaseModel):
    status: str
    version: str = "1.0.0"
    checks: dict[str, str] = Field(default_factory=dict)