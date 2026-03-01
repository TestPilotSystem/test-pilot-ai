from pydantic_settings import BaseSettings, SettingsConfigDict
from functools import lru_cache


class Settings(BaseSettings):
    
    llm_model: str = "gemma2:9b"
    llm_temperature: float = 0.7
    
    llm_provider: str = "ollama"
    groq_model: str = "llama-3.1-8b-instant"
    groq_api_key: str = ""
    
    chunk_size: int = 1000
    chunk_overlap: int = 200
    default_search_k: int = 4
    embeddings_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    
    cors_origins: str = "http://localhost:3000"
    upload_dir: str = "temp_uploads"
    
    chroma_path: str = "vector_db"
    
    database_url: str = "mysql+pymysql://root:rootpassword@localhost:3308/testpilotdb"
    
    model_config = SettingsConfigDict(env_file=".env", case_sensitive=False)


@lru_cache()
def get_settings() -> Settings:
    return Settings()


settings = get_settings()

