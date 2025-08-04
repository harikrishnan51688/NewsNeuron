from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application configuration settings."""

    POSTGRES_URL: str = "postgresql://postgres:password@localhost:5432/newsneuron"


    OPENROUTER_API_KEY: str = "your_openrouter_api_key"
    OPENROUTER_BASE_URL: str = "https://api.openrouter.ai/v1"


settings = Settings()
