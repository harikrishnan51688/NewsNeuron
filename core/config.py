import os
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    # Database URLs
    POSTGRES_URL: str = "postgresql://user:password@localhost:5432/newsneuron"
    NEO4J_URL: str = "bolt://localhost:7687"
    NEO4J_USER: str = "neo4j"
    NEO4J_PASSWORD: str = "password"
    REDIS_URL: str = "redis://localhost:6379"
    
    # API Keys
    OPENAI_API_KEY: str = ""
    NEWS_API_KEY: str = ""  # For NewsAPI.org
    
    # Embedding settings
    EMBEDDING_MODEL: str = "text-embedding-3-small"
    EMBEDDING_DIMENSION: int = 1536
    
    # Retrieval settings
    MAX_RESULTS: int = 50
    SIMILARITY_THRESHOLD: float = 0.7
    
    class Config:
        env_file = ".env"

settings = Settings()