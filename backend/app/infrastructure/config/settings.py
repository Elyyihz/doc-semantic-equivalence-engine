from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application runtime configurations loaded via environment variables."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    APP_NAME: str = "doc-semantic-equivalence-engine"
    APP_VERSION: str = "1.0.0"
    ENVIRONMENT: str = "production"
    LOG_LEVEL: str = "INFO"

    # Vector Database (Qdrant)
    QDRANT_HOST: str = "qdrant"
    QDRANT_PORT: int = 6333
    QDRANT_COLLECTION: str = "document_embeddings"

    # LLM & Embedding Runtimes (vLLM / Local)
    VLLM_API_BASE: str = "http://vllm-runtime:8001/v1"
    VLLM_API_KEY: str = "EMPTY"
    VLLM_MODEL_NAME: str = "Qwen/Qwen2.5-7B-Instruct"
    EMBEDDING_MODEL_NAME: str = "BAAI/bge-m3"

    # OpenTelemetry
    OTEL_EXPORTER_OTLP_ENDPOINT: str = "http://otel-collector:4317"
    OTEL_SERVICE_NAME: str = "doc-semantic-equivalence-engine"


settings = Settings()
