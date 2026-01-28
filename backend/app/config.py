"""Configuration settings for Kernle backend."""

from functools import lru_cache

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings loaded from environment."""

    # Supabase
    supabase_url: str
    # New key system (preferred)
    supabase_secret_key: str | None = None  # Backend/admin access
    supabase_publishable_key: str | None = None  # Client/public access
    # Legacy keys (deprecated, will be removed)
    supabase_service_role_key: str | None = None
    supabase_anon_key: str | None = None
    database_url: str | None = None  # Optional - for direct Postgres access

    # JWT
    jwt_secret_key: str  # Required - no default for security
    jwt_algorithm: str = "HS256"
    jwt_expire_minutes: int = 60 * 24 * 7  # 1 week

    # App
    debug: bool = False
    # CORS: Allowed origins for cross-origin requests
    cors_origins: list[str] = [
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "https://kernle.ai",
        "https://www.kernle.ai",
    ]

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"  # Ignore extra env vars not in model


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()
