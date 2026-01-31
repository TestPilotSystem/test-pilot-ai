from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    """
    Configuración centralizada de la aplicación.
    Los valores se cargan desde variables de entorno o archivo .env
    """
    
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
    
    class Config:
        env_file = ".env"
        case_sensitive = False


@lru_cache()
def get_settings() -> Settings:
    """
    Retorna una instancia cacheada de la configuración.
    Usar esta función para acceder a la configuración en toda la app.
    """
    return Settings()


settings = get_settings()
