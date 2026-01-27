"""Configuration settings for Kernle backend."""

from functools import lru_cache
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings loaded from environment."""
    
    # Supabase
    supabase_url: str
    supabase_service_role_key: str
    supabase_anon_key: str
    database_url: str | None = None  # Optional - for direct Postgres access
    
    # JWT
    jwt_secret_key: str = "kernle-dev-secret-change-in-production"
    jwt_algorithm: str = "HS256"
    jwt_expire_minutes: int = 60 * 24 * 7  # 1 week
    
    # App
    debug: bool = False
    cors_origins: list[str] = ["*"]
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"  # Ignore extra env vars not in model


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()
