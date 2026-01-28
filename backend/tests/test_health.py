"""Test health endpoints."""


def test_root(client):
    """Test root endpoint returns service info."""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["service"] == "kernle-backend"
    assert data["status"] == "ok"


def test_health(client):
    """Test health endpoint returns valid status."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    # Status is "healthy" with DB, "degraded" without - both are valid responses
    assert data["status"] in ["healthy", "degraded"]
    # Database field should exist and describe connection state
    assert "database" in data
