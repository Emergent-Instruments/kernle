"""Tests for server-side re-embedding on sync."""

from datetime import datetime, timezone
from unittest.mock import AsyncMock, patch

import pytest


class TestServerSideReembedding:
    """Test that sync push always re-embeds server-side."""

    @pytest.fixture
    def client(self):
        """Create test client."""
        from app.main import app
        from fastapi.testclient import TestClient

        return TestClient(app)

    @pytest.fixture
    def auth_headers(self):
        """Create auth headers with a test token."""
        from app.auth import create_access_token
        from app.config import get_settings

        settings = get_settings()
        token = create_access_token(settings, user_id="usr_test123456", agent_id="test-agent")
        return {"Authorization": f"Bearer {token}"}

    @patch("app.routes.sync.upsert_memory", new_callable=AsyncMock)
    @patch("app.routes.sync.update_agent_last_sync", new_callable=AsyncMock)
    @patch("app.routes.sync.get_agent_by_user_and_name", new_callable=AsyncMock)
    @patch("app.routes.sync.create_embedding", new_callable=AsyncMock)
    def test_push_ignores_client_embedding(
        self,
        mock_create_embedding,
        mock_get_agent,
        mock_update_sync,
        mock_upsert,
        client,
        auth_headers,
    ):
        """Client's 384-dim embedding should be stripped and replaced with server 1536-dim."""
        # Setup: server returns 1536-dim embedding
        mock_create_embedding.return_value = [0.1] * 1536
        mock_get_agent.return_value = {"id": "agent-uuid"}
        mock_upsert.return_value = None
        mock_update_sync.return_value = None

        # Client sends 384-dim embedding (should be ignored)
        response = client.post(
            "/sync/push",
            headers=auth_headers,
            json={
                "operations": [
                    {
                        "operation": "insert",
                        "table": "notes",
                        "record_id": "test-note-1",
                        "data": {
                            "content": "Test note content",
                            "note_type": "note",
                            "embedding": [0.5] * 384,  # Client's 384-dim embedding
                        },
                        "local_updated_at": datetime.now(timezone.utc).isoformat(),
                        "version": 1,
                    }
                ]
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert data["synced"] == 1

        # Verify upsert was called with server's 1536-dim embedding, not client's
        mock_upsert.assert_called_once()
        call_args = mock_upsert.call_args
        # upsert_memory(db, agent_id, table, record_id, data, agent_ref=...)
        stored_data = call_args[0][4]  # 5th positional arg is data dict

        # Should have 1536-dim embedding, not 384
        assert "embedding" in stored_data
        assert len(stored_data["embedding"]) == 1536

    @patch("app.routes.sync.upsert_memory", new_callable=AsyncMock)
    @patch("app.routes.sync.update_agent_last_sync", new_callable=AsyncMock)
    @patch("app.routes.sync.get_agent_by_user_and_name", new_callable=AsyncMock)
    @patch("app.routes.sync.create_embedding", new_callable=AsyncMock)
    def test_push_without_client_embedding(
        self,
        mock_create_embedding,
        mock_get_agent,
        mock_update_sync,
        mock_upsert,
        client,
        auth_headers,
    ):
        """Push works even when client doesn't send any embedding."""
        mock_create_embedding.return_value = [0.1] * 1536
        mock_get_agent.return_value = {"id": "agent-uuid"}
        mock_upsert.return_value = None
        mock_update_sync.return_value = None

        response = client.post(
            "/sync/push",
            headers=auth_headers,
            json={
                "operations": [
                    {
                        "operation": "insert",
                        "table": "notes",
                        "record_id": "test-note-2",
                        "data": {
                            "content": "Note without embedding",
                            "note_type": "note",
                            # No embedding field
                        },
                        "local_updated_at": datetime.now(timezone.utc).isoformat(),
                        "version": 1,
                    }
                ]
            },
        )

        assert response.status_code == 200
        assert response.json()["synced"] == 1

        # Should still generate 1536-dim embedding
        mock_upsert.assert_called_once()
        stored_data = mock_upsert.call_args[0][4]  # 5th positional arg is data dict
        assert len(stored_data["embedding"]) == 1536

    @patch("app.routes.sync.upsert_memory", new_callable=AsyncMock)
    @patch("app.routes.sync.update_agent_last_sync", new_callable=AsyncMock)
    @patch("app.routes.sync.get_agent_by_user_and_name", new_callable=AsyncMock)
    @patch("app.routes.sync.create_embedding", new_callable=AsyncMock)
    def test_push_with_failed_embedding(
        self,
        mock_create_embedding,
        mock_get_agent,
        mock_update_sync,
        mock_upsert,
        client,
        auth_headers,
    ):
        """If server embedding fails, don't store client's embedding either."""
        mock_create_embedding.return_value = None  # Embedding failed
        mock_get_agent.return_value = {"id": "agent-uuid"}
        mock_upsert.return_value = None
        mock_update_sync.return_value = None

        response = client.post(
            "/sync/push",
            headers=auth_headers,
            json={
                "operations": [
                    {
                        "operation": "insert",
                        "table": "notes",
                        "record_id": "test-note-3",
                        "data": {
                            "content": "Note content",
                            "note_type": "note",
                            "embedding": [0.5] * 384,  # Client's embedding
                        },
                        "local_updated_at": datetime.now(timezone.utc).isoformat(),
                        "version": 1,
                    }
                ]
            },
        )

        assert response.status_code == 200
        assert response.json()["synced"] == 1

        # Stored data should NOT have embedding (client's was stripped, server's failed)
        mock_upsert.assert_called_once()
        stored_data = mock_upsert.call_args[0][4]  # 5th positional arg is data dict
        assert "embedding" not in stored_data


class TestPullStripsEmbeddings:
    """Test that pull responses don't include embeddings."""

    @pytest.fixture
    def client(self):
        from app.main import app
        from fastapi.testclient import TestClient

        return TestClient(app)

    @pytest.fixture
    def auth_headers(self):
        from app.auth import create_access_token
        from app.config import get_settings

        settings = get_settings()
        token = create_access_token(settings, user_id="usr_test123456", agent_id="test-agent")
        return {"Authorization": f"Bearer {token}"}

    @patch("app.routes.sync.get_changes_since", new_callable=AsyncMock)
    def test_pull_strips_embeddings(self, mock_get_changes, client, auth_headers):
        """Pull response should not include embeddings (saves bandwidth)."""
        # Simulate DB returning records with 1536-dim embeddings
        # get_changes_since returns (changes, has_more) tuple
        mock_get_changes.return_value = (
            [
                {
                    "operation": "update",
                    "table": "notes",
                    "record_id": "note-1",
                    "data": {
                        "content": "Note content",
                        "note_type": "note",
                        "embedding": [0.1] * 1536,  # Server's 1536-dim embedding
                        "created_at": datetime.now(timezone.utc).isoformat(),
                    },
                }
            ],
            False,  # has_more
        )

        response = client.post("/sync/pull", headers=auth_headers, json={})

        assert response.status_code == 200
        data = response.json()
        assert len(data["operations"]) == 1

        # Embedding should be stripped from response
        operation_data = data["operations"][0]["data"]
        assert "embedding" not in operation_data
        assert operation_data["content"] == "Note content"
