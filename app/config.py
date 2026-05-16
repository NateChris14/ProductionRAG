from pydantic_settings import BaseSettings, SettingsConfigDict
from functools import lru_cache

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # App
    app_env: str = "development"
    log_level: str = "INFO"
    secret_key: str = "changeme"

    # Postgres
    database_url: str = "postgresql+asyncpg://rag:secret@localhost:5432/ragdb"

    # Redis
    redis_url: str = "redis://localhost:6379/0"
    cache_ttl_seconds: int = 300

    # Vector Backend
    vector_backend: str = "milvus"
    milvus_uri: str = "http://localhost:19530"
    milvus_token: str = ""
    pinecone_api_key: str = ""
    pinecone_environment: str = "us-east-1"

    # Embeddings
    embed_provider: str = "openai"
    embed_model: str = "text-embedding-3-small"
    embed_dims: int = 1536
    embed_batch_size: int = 128
    openai_api_key: str = ""

     # LLM
    llm_strong: str = "gpt-4o"
    llm_fast: str = "gpt-4o-mini"
    anthropic_api_key: str = ""
    llm_stream: bool = True

    # Reranker
    rerank_provider: str = "flashrank"
    cohere_api_key: str = ""
    rerank_top_n: int = 8

    # Retrieval
    dense_top_k: int = 80
    sparse_top_k: int = 80
    rerank_candidates: int = 40
    max_context_tokens: int = 2200

    # Observability
    langfuse_public_key: str = ""
    langfuse_secret_key: str = ""
    langfuse_host: str = "https://cloud.langfuse.com"
    otel_exporter_otlp_endpoint: str = "http://localhost:4317"

@lru_cache
def get_settings() -> Settings:
    return Settings()


