"""Pytest configuration and fixtures."""

import os
import secrets
import sys
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# Generate a unique test secret for this test run to prevent token forgery
_TEST_JWT_SECRET = f"test-only-{secrets.token_urlsafe(32)}"

# For unit tests, set mock values ONLY if not running integration tests
if not os.environ.get("RUN_INTEGRATION"):
    os.environ.setdefault("SUPABASE_URL", "https://test.supabase.co")
    os.environ.setdefault("SUPABASE_SECRET_KEY", "test-secret-key")  # New key system
    os.environ.setdefault("SUPABASE_PUBLISHABLE_KEY", "test-publishable-key")
    # Legacy keys for backwards compatibility
    os.environ.setdefault("SUPABASE_SERVICE_ROLE_KEY", "test-service-key")
    os.environ.setdefault("SUPABASE_ANON_KEY", "test-anon-key")
    os.environ.setdefault("DATABASE_URL", "postgresql://test:test@localhost/test")
    os.environ.setdefault("JWT_SECRET_KEY", _TEST_JWT_SECRET)
else:
    # For integration tests, load from .env
    from pathlib import Path

    from dotenv import load_dotenv

    env_path = Path(__file__).parent.parent / ".env"

    # Safety check: require explicit confirmation for integration tests
    if not os.environ.get("CONFIRM_INTEGRATION_CREDENTIALS"):
        print(
            "\n" + "=" * 70,
            file=sys.stderr,
        )
        print(
            "⚠️  WARNING: Integration tests will use REAL credentials from .env",
            file=sys.stderr,
        )
        print(
            "   This may affect production databases and consume API quotas.",
            file=sys.stderr,
        )
        print(
            "   Set CONFIRM_INTEGRATION_CREDENTIALS=yes to proceed.",
            file=sys.stderr,
        )
        print("=" * 70 + "\n", file=sys.stderr)
        pytest.exit(
            "Integration tests require CONFIRM_INTEGRATION_CREDENTIALS=yes",
            returncode=1,
        )

    print(
        "\n⚠️  Running integration tests with REAL credentials\n",
        file=sys.stderr,
    )
    load_dotenv(env_path, override=True)

from app.main import app  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402


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
    # Use clearly invalid test ID that cannot collide with production IDs
    token = create_access_token(settings, user_id="usr_TEST_ONLY_000000")
    return {"Authorization": f"Bearer {token}"}


@pytest.fixture(autouse=True)
def mock_user_lookup(monkeypatch):
    """Avoid real Supabase calls for auth in unit tests."""
    if os.environ.get("RUN_INTEGRATION"):
        return

    async def _fake_get_user(db, user_id):
        return {"user_id": user_id, "tier": "free", "is_admin": False}

    monkeypatch.setattr("app.database.get_user", _fake_get_user)

    async def _fake_ensure_agent_exists(db, user_id, agent_id):
        return {"id": "agent-uuid", "agent_id": agent_id, "user_id": user_id}

    monkeypatch.setattr("app.database.ensure_agent_exists", _fake_ensure_agent_exists)
    monkeypatch.setattr("app.routes.sync.ensure_agent_exists", _fake_ensure_agent_exists)
