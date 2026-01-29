"""Integration tests - run against real Supabase when available.

Run with: pytest tests/test_integration.py -v --run-integration
Skip with: pytest tests/ -v (default - skips integration tests)
"""

import os

import pytest

# Skip all tests in this file unless --run-integration is passed
pytestmark = pytest.mark.skipif(
    not os.environ.get("RUN_INTEGRATION"), reason="Integration tests require RUN_INTEGRATION=1"
)


@pytest.fixture
def real_client():
    """Create a test client with real Supabase connection.

    Uses the shared client from conftest which loads real env when RUN_INTEGRATION=1.
    """
    # Reset supabase client to ensure fresh connection
    from app import database
    from app.main import app
    from fastapi.testclient import TestClient

    database._supabase_client = None

    return TestClient(app)


class TestIntegration:
    """Integration tests against real Supabase."""

    def test_health_with_db(self, real_client):
        """Test health endpoint with real DB."""
        response = real_client.get("/health")
        assert response.status_code == 200

    def test_register_and_login(self, real_client):
        """Test full registration and login flow."""
        import uuid

        agent_id = f"test-agent-{uuid.uuid4().hex[:8]}"

        # Register
        response = real_client.post("/auth/register", json={"agent_id": agent_id})
        assert response.status_code == 200
        data = response.json()
        assert "access_token" in data

        # TODO: Get secret from response header and test login
        # This requires updating the register endpoint to return the secret

        # Me endpoint
        headers = {"Authorization": f"Bearer {data['access_token']}"}
        response = real_client.get("/auth/me", headers=headers)
        assert response.status_code == 200
        me_data = response.json()
        assert me_data["agent_id"] == agent_id

    def test_sync_push_pull(self, real_client):
        """Test sync push and pull flow."""
        import uuid
        from datetime import datetime, timezone

        # First register
        agent_id = f"test-sync-{uuid.uuid4().hex[:8]}"
        response = real_client.post("/auth/register", json={"agent_id": agent_id})
        token = response.json()["access_token"]
        headers = {"Authorization": f"Bearer {token}"}

        # Push a note
        response = real_client.post(
            "/sync/push",
            headers=headers,
            json={
                "operations": [
                    {
                        "operation": "insert",
                        "table": "notes",
                        "record_id": f"note-{uuid.uuid4().hex[:8]}",
                        "data": {"content": "Test note from integration test", "note_type": "note"},
                        "local_updated_at": datetime.now(timezone.utc).isoformat(),
                        "version": 1,
                    }
                ]
            },
        )
        assert response.status_code == 200
        push_data = response.json()
        assert push_data["synced"] == 1

        # Pull changes
        response = real_client.post("/sync/pull", headers=headers, json={})
        assert response.status_code == 200
        pull_data = response.json()
        assert len(pull_data["operations"]) >= 1

    def test_memory_search(self, real_client):
        """Test memory search."""
        import uuid

        # Register
        agent_id = f"test-search-{uuid.uuid4().hex[:8]}"
        response = real_client.post("/auth/register", json={"agent_id": agent_id})
        token = response.json()["access_token"]
        headers = {"Authorization": f"Bearer {token}"}

        # Search (should return empty for new agent)
        response = real_client.post("/memories/search", headers=headers, json={"query": "test"})
        assert response.status_code == 200
        data = response.json()
        assert "results" in data
        assert data["query"] == "test"
