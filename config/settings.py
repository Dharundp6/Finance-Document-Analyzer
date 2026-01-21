"""
Application configuration settings.
"""

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings with validation."""

    # API Keys
    google_api_key: str = Field(..., env="GOOGLE_API_KEY")

    # Application Settings
    app_name: str = Field(default="Financial RAG System", env="APP_NAME")
    app_version: str = "1.0.0"
    debug: bool = Field(default=False, env="DEBUG")

    # Document Processing
    max_file_size_mb: int = Field(default=50, env="MAX_FILE_SIZE_MB")
    supported_extensions: list = [".pdf", ".docx", ".xlsx", ".csv", ".txt"]

    # Chunking Configuration
    chunk_size: int = Field(default=1000, env="CHUNK_SIZE")
    chunk_overlap: int = Field(default=200, env="CHUNK_OVERLAP")
    min_chunk_size: int = Field(default=100, env="MIN_CHUNK_SIZE")

    # Embedding Configuration - Local Sentence Transformers
    embedding_model: str = Field(
        default="all-MiniLM-L6-v2",  # Fast and good quality
        env="EMBEDDING_MODEL",
    )
    embedding_dimension: int = Field(default=384, env="EMBEDDING_DIMENSION")  # MiniLM dimension
    embedding_batch_size: int = Field(default=32, env="EMBEDDING_BATCH_SIZE")  # Can be larger now!

    # LLM Configuration - Using Gemini Flash
    llm_model: str = Field(default="gemini-1.5-flash-latest", env="LLM_MODEL")
    temperature: float = Field(default=0.1, env="TEMPERATURE")
    max_output_tokens: int = Field(default=2048, env="MAX_OUTPUT_TOKENS")

    # FAISS Vector Store Configuration
    vector_db_path: str = Field(default="./data/vector_db", env="VECTOR_DB_PATH")
    faiss_index_file: str = Field(default="faiss_index.bin", env="FAISS_INDEX_FILE")
    metadata_file: str = Field(default="metadata.pkl", env="METADATA_FILE")

    # Retrieval Configuration
    top_k: int = Field(default=10, env="TOP_K")
    similarity_threshold: float = Field(default=0.3, env="SIMILARITY_THRESHOLD")
    rerank_top_k: int = Field(default=5, env="RERANK_TOP_K")

    # Rate Limiting (only for Gemini LLM now)
    rate_limit_delay: float = Field(default=1.0, env="RATE_LIMIT_DELAY")
    retry_base_delay: float = Field(default=30.0, env="RETRY_BASE_DELAY")

    # File Paths
    upload_dir: str = Field(default="./data/uploads", env="UPLOAD_DIR")
    processed_dir: str = Field(default="./data/processed", env="PROCESSED_DIR")

    # API Configuration
    api_host: str = Field(default="0.0.0.0", env="API_HOST")
    api_port: int = Field(default=8000, env="API_PORT")

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()


settings = get_settings()
