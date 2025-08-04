from pydantic_settings import BaseSettings
import os
from dotenv import load_dotenv

load_dotenv()


class Settings(BaseSettings):
    """Application configuration settings."""

    POSTGRES_URL: str = "postgresql://postgres:password@localhost:5432/newsneuron"


    OPENROUTER_API_KEY: str = os.getenv("OPENROUTER_API_KEY")
    OPENROUTER_BASE_URL: str = "https://api.openrouter.ai/v1"

    EMBEDDING_MODEL: str = "all-MiniLM-L6-v2" # Local model for embeddings from Sentence Transformers
    EMBEDDING_DIMENSION: int = 384
    
    PINECONE_API_KEY: str = os.getenv("PINECONE_API_KEY")
    PINECONE_ENVIRONMENT: str = os.getenv("PINECONE_ENVIRONMENT")  
    PINECONE_INDEX_NAME: str = "news-articles"

settings = Settings()
