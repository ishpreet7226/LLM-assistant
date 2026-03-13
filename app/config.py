from pydantic_settings import BaseSettings
from pathlib import Path


class Settings(BaseSettings):
    # Ollama
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    OLLAMA_MODEL: str = "llama3.3"
    OLLAMA_TIMEOUT: int = 120

    # Embeddings
    EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"
    EMBEDDING_DIMENSION: int = 384

    # Vector Store
    VECTOR_STORE_PATH: str = "vector_store/faiss_index"
    CHUNK_SIZE: int = 512
    CHUNK_OVERLAP: int = 64
    TOP_K_RESULTS: int = 5

    # Paths
    DOCUMENTS_DIR: str = "data/documents"

    # RAG
    MAX_CONTEXT_TOKENS: int = 3000
    SYSTEM_PROMPT: str = (
        "You are a helpful assistant that answers questions based on the provided context. "
        "If the context doesn't contain enough information to answer confidently, say so. "
        "Be concise, accurate, and cite relevant parts of the context when appropriate."
    )

    class Config:
        env_file = ".env"
        extra = "ignore"


settings = Settings()
