from pydantic_settings import BaseSettings, SettingsConfigDict
from functools import lru_cache


class Settings(BaseSettings):

    # LLM - Ollama (local)
    llm_model: str = "gemma2:9b"
    llm_temperature: float = 0.7

    # LLM - Groq (cloud)
    llm_provider: str = "ollama"
    groq_model: str = "llama-3.3-70b-versatile"
    groq_api_key: str = ""

    # RAG - Retrieval
    default_search_k: int = 8
    fetch_k_multiplier: int = 4

    # RAG - Parent-child chunking
    parent_chunk_size: int = 1500
    child_chunk_size: int = 350
    child_chunk_overlap: int = 50

    # Legacy chunk settings (kept for backward compat with existing DBs)
    chunk_size: int = 1000
    chunk_overlap: int = 200

    # Embeddings & reranking
    embeddings_model: str = "intfloat/multilingual-e5-large"
    rerank_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    cors_origins: str = "http://localhost:3000"
    upload_dir: str = "temp_uploads"

    chroma_path: str = "vector_db"
    database_url: str = "mysql+pymysql://root:rootpassword@localhost:3308/testpilotdb"

    model_config = SettingsConfigDict(env_file=".env", case_sensitive=False)


@lru_cache()
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
