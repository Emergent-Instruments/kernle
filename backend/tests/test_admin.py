"""Test admin endpoints."""

import os

import pytest


class TestAdminEndpoints:
    """Test admin API endpoints."""

    def test_stats_requires_auth(self, client):
        """Test that stats endpoint requires authentication."""
        response = client.get("/admin/stats")
        assert response.status_code == 401

    def test_agents_requires_auth(self, client):
        """Test that agents list requires authentication."""
        response = client.get("/admin/agents")
        assert response.status_code == 401

    def test_backfill_requires_auth(self, client):
        """Test that backfill endpoint requires authentication."""
        response = client.post("/admin/embeddings/backfill", json={"agent_id": "test"})
        assert response.status_code == 401


class TestAdminIntegration:
    """Integration tests for admin endpoints (requires real DB)."""

    @pytest.mark.skipif(
        not os.environ.get("RUN_INTEGRATION"), reason="Integration tests require RUN_INTEGRATION=1"
    )
    def test_stats_with_auth(self, client, auth_headers):
        """Test stats endpoint with valid auth."""
        response = client.get("/admin/stats", headers=auth_headers)
        # Should pass auth (may fail on DB)
        assert response.status_code != 401

    @pytest.mark.skipif(
        not os.environ.get("RUN_INTEGRATION"), reason="Integration tests require RUN_INTEGRATION=1"
    )
    def test_agents_with_auth(self, client, auth_headers):
        """Test agents list with valid auth."""
        response = client.get("/admin/agents", headers=auth_headers)
        assert response.status_code != 401
