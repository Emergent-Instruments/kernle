"""Pytest configuration and fixtures."""

import pytest
from fastapi.testclient import TestClient

# Mock the settings before importing app
import os
os.environ.setdefault("SUPABASE_URL", "https://test.supabase.co")
os.environ.setdefault("SUPABASE_SERVICE_ROLE_KEY", "test-service-key")
os.environ.setdefault("SUPABASE_ANON_KEY", "test-anon-key")
os.environ.setdefault("DATABASE_URL", "postgresql://test:test@localhost/test")
os.environ.setdefault("JWT_SECRET_KEY", "test-secret-key-for-testing-only")

from app.main import app


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
