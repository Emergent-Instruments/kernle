"""Pytest configuration and fixtures."""

import os

import pytest

# For unit tests, set mock values ONLY if not running integration tests
if not os.environ.get("RUN_INTEGRATION"):
    os.environ.setdefault("SUPABASE_URL", "https://test.supabase.co")
    os.environ.setdefault("SUPABASE_SECRET_KEY", "test-secret-key")  # New key system
    os.environ.setdefault("SUPABASE_PUBLISHABLE_KEY", "test-publishable-key")
    # Legacy keys for backwards compatibility
    os.environ.setdefault("SUPABASE_SERVICE_ROLE_KEY", "test-service-key")
    os.environ.setdefault("SUPABASE_ANON_KEY", "test-anon-key")
    os.environ.setdefault("DATABASE_URL", "postgresql://test:test@localhost/test")
    os.environ.setdefault("JWT_SECRET_KEY", "test-secret-key-for-testing-only")
else:
    # For integration tests, load from .env
    from pathlib import Path

    from dotenv import load_dotenv
    env_path = Path(__file__).parent.parent / ".env"
    load_dotenv(env_path, override=True)

from app.main import app
from fastapi.testclient import TestClient


@pytest.fixture
def client():
    """Create a test client."""
    return TestClient(app)


@pytest.fixture
def auth_headers():
    """Create auth headers with a test token."""
    from app.auth import create_access_token
    from app.config import get_settings

    settings = get_settings()
    token = create_access_token("test-agent", settings)
    return {"Authorization": f"Bearer {token}"}
