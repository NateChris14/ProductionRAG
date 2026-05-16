"""
SQLAlchemy ORM models. Documents, chunks, and ingestion jobs are tracked in
Postgres. Vectors live in Milvus/Pinecone; Postgres stores metadata only.
"""

from sqlalchemy import Column, String, Integer, Text, DateTime, Boolean, JSON, ForeignKey, Index
from sqlalchemy.orm import declarative_base, relationship
from sqlalchemy.sql import func

Base = declarative_base()

class Document(Base):
    __tablename__ = "documents"

    id = Column(String(36), primary_key=True)
    tenant_id = Column(String(128), nullable=False, index=True)
    title = Column(Text, nullable=True)
    source_type = Column(String(64), nullable=False, default="text")
    source_uri = Column(Text, nullable=True)
    content_hash = Column(String(64), nullable=False) # SHA-256 of content
    acl_tags = Column(JSON, default=list)
    extra_meta = Column(JSON, default=dict)
    is_active = Column(Boolean, default=True, index=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

    # Relationships
    chunks = relationship("DocumentChunk", back_populates="document",
    cascade="all, delete-orphan")
    jobs = relationship("IngestionJob", back_populates="document")

    __table_args__ = (
        Index("ix_doc_tenant_active", "tenant_id", "is_active"),
    )

class DocumentChunk(Base):
    __tablename__ = "document_chunks"

    id = Column(String(36), primary_key=True)
    document_id = Column(String(36), ForeignKey("documents.id"), nullable=False, index=True)
    tenant_id = Column(String(128), nullable=False, index=True)
    chunk_index = Column(Integer, nullable=False)
    parent_id = Column(String(36), nullable=True) # parent chunk for hierarchical chunks
    text = Column(Text, nullable=False)
    token_count = Column(Integer, nullable=False, default=0)
    vector_id = Column(String(64), nullable=True) # vector ID in Milvus/Pinecone
    embed_model = Column(String(128), nullable=True)
    acl_tags = Column(JSON, default=list)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    document = relationship("Document", back_populates="chunks")

    __table_args__ = (
        Index("ix_chunk_tenant_doc", "tenant_id", "document_id"),
    )

class IngestionJob(Base):
    __tablename__ = "ingestion_jobs"

    id = Column(String(36), primary_key=True)
    document_id = Column(String(36), ForeignKey("documents.id"), nullable=False)
    tenant_id = Column(String(128), nullable=False)
    status = Column(String(32), default="queued") # queued, running, done, failed
    error = Column(Text, nullable=True)
    stages = Column(JSON, default=dict) # per-stage timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

    document = relationship("Document", back_populates="jobs")

    


