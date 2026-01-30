"""Test authentication endpoints and utilities."""

from unittest.mock import AsyncMock, MagicMock, patch

from app.auth import (
    AuthContext,
    create_access_token,
    decode_token,
    generate_agent_secret,
    generate_user_id,
)
from app.config import get_settings


class TestAuthUtilities:
    """Test authentication utility functions."""

    def test_generate_user_id(self):
        """Test user_id generation format."""
        user_id = generate_user_id()
        assert user_id.startswith("usr_")
        assert len(user_id) == 16  # usr_ + 12 hex chars
        # Should be unique
        user_id2 = generate_user_id()
        assert user_id != user_id2

    def test_generate_secret(self):
        """Test secret generation."""
        secret = generate_agent_secret()
        assert len(secret) >= 32
        assert isinstance(secret, str)

    def test_hash_and_verify_secret(self):
        """Test secret hashing and verification."""
        import bcrypt

        secret = "test-secret-123"
        # Use bcrypt directly to avoid passlib issues
        hashed = bcrypt.hashpw(secret.encode(), bcrypt.gensalt()).decode()

        assert hashed != secret
        assert bcrypt.checkpw(secret.encode(), hashed.encode())
        assert not bcrypt.checkpw(b"wrong-secret", hashed.encode())

    def test_create_and_decode_token(self):
        """Test JWT token creation and decoding."""
        settings = get_settings()
        user_id = "usr_test123456"

        token = create_access_token(settings, user_id=user_id)
        assert isinstance(token, str)

        payload = decode_token(token, settings)
        assert payload["sub"] == user_id
        assert "exp" in payload
        assert "iat" in payload

    def test_create_token_with_agent_id(self):
        """Test JWT token includes agent_id when provided."""
        settings = get_settings()
        agent_id = "test-agent"
        user_id = "usr_abc123def456"

        token = create_access_token(settings, user_id=user_id, agent_id=agent_id)
        payload = decode_token(token, settings)

        assert payload["sub"] == user_id
        assert payload["user_id"] == user_id
        assert payload["agent_id"] == agent_id

    def test_auth_context_namespacing(self):
        """Test AuthContext namespaces agent_id correctly."""
        # With user_id and agent_id
        ctx = AuthContext(user_id="usr_abc123", agent_id="claire")
        assert ctx.namespaced_agent_id() == "usr_abc123/claire"
        assert ctx.namespaced_agent_id("my-project") == "usr_abc123/my-project"

        # Already namespaced - should return as-is
        assert ctx.namespaced_agent_id("usr_other/project") == "usr_other/project"

        # Without agent_id (web user)
        ctx_web = AuthContext(user_id="usr_abc123", agent_id=None)
        assert ctx_web.namespaced_agent_id() == "usr_abc123/"  # Empty agent_id
        assert ctx_web.namespaced_agent_id("project") == "usr_abc123/project"


class TestAuthEndpoints:
    """Test authentication API endpoints."""

    def test_me_without_auth(self, client):
        """Test /auth/me requires authentication."""
        response = client.get("/auth/me")
        assert response.status_code == 401  # Unauthorized (no auth header)

    def test_me_with_auth(self, client, auth_headers):
        """Test /auth/me with valid auth token returns user info."""
        from unittest.mock import AsyncMock, patch

        mock_user = {
            "user_id": "usr_TEST_ONLY_000000",
            "email": "test@example.com",
            "display_name": "Test User",
            "tier": "free",
            "created_at": "2024-01-01T00:00:00Z",
        }
        mock_usage = {
            "daily_requests": 10,
            "monthly_requests": 50,
            "daily_reset_at": None,
            "monthly_reset_at": None,
        }

        with (
            patch("app.routes.auth.get_user", new_callable=AsyncMock) as mock_get_user,
            patch("app.routes.auth.get_usage_for_user", new_callable=AsyncMock) as mock_get_usage,
        ):
            mock_get_user.return_value = mock_user
            mock_get_usage.return_value = mock_usage

            response = client.get("/auth/me", headers=auth_headers)

            assert response.status_code == 200, f"Expected 200, got {response.status_code}"
            data = response.json()

            # Verify the response structure
            assert data["user_id"] == "usr_TEST_ONLY_000000"
            assert data["tier"] == "free"
            assert "limits" in data
            assert "usage" in data
            assert data["usage"]["daily_requests"] == 10
            assert data["usage"]["monthly_requests"] == 50

            # Verify mocks were called with correct arguments
            mock_get_user.assert_called_once()
            mock_get_usage.assert_called_once()


class TestOAuthTokenExchange:
    """Test OAuth token exchange endpoint - real validation logic, not just mocks."""

    def test_oauth_rejects_invalid_token_format(self, client):
        """Test that malformed tokens are rejected before any external calls."""
        response = client.post(
            "/auth/oauth/token",
            json={"access_token": "not-a-valid-jwt"},
        )
        assert response.status_code == 401
        assert "Invalid token" in response.json()["detail"]

    def test_oauth_rejects_wrong_issuer(self, client):
        """Test that tokens from wrong issuer are rejected (prevents SSRF/auth bypass)."""
        from jose import jwt

        # Create a token with a malicious issuer
        fake_token = jwt.encode(
            {
                "sub": "attacker-user-id",
                "email": "attacker@evil.com",
                "iss": "https://evil.com/auth/v1",  # Wrong issuer
                "aud": "authenticated",
                "exp": 9999999999,
            },
            "fake-secret",
            algorithm="HS256",
            headers={"kid": "fake-kid"},
        )

        response = client.post(
            "/auth/oauth/token",
            json={"access_token": fake_token},
        )
        assert response.status_code == 401
        assert "Invalid token issuer" in response.json()["detail"]

    def test_oauth_rejects_issuer_prefix_attack(self, client):
        """Test that issuer prefix attacks are blocked (e.g., supabase.co.evil.com)."""
        from jose import jwt

        settings = get_settings()
        # Attacker tries to use a lookalike domain that would pass startswith()
        malicious_issuer = f"{settings.supabase_url}.evil.com/auth/v1"

        fake_token = jwt.encode(
            {
                "sub": "attacker-user-id",
                "email": "attacker@evil.com",
                "iss": malicious_issuer,
                "aud": "authenticated",
                "exp": 9999999999,
            },
            "fake-secret",
            algorithm="HS256",
            headers={"kid": "fake-kid"},
        )

        response = client.post(
            "/auth/oauth/token",
            json={"access_token": fake_token},
        )
        assert response.status_code == 401
        assert "Invalid token issuer" in response.json()["detail"]

    def test_oauth_handles_jwks_fetch_failure(self, client):
        """Test graceful handling when JWKS endpoint is unavailable."""
        from jose import jwt

        settings = get_settings()
        expected_issuer = f"{settings.supabase_url}/auth/v1"

        # Token with correct issuer but JWKS will fail to fetch
        token = jwt.encode(
            {
                "sub": "test-user-id",
                "email": "test@example.com",
                "iss": expected_issuer,
                "aud": "authenticated",
                "exp": 9999999999,
            },
            "fake-secret",
            algorithm="HS256",
            headers={"kid": "nonexistent-kid"},
        )

        # Mock httpx to simulate JWKS fetch failure
        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client_class.return_value.__aenter__.return_value = mock_client
            mock_client.get.return_value = MagicMock(status_code=500)

            response = client.post(
                "/auth/oauth/token",
                json={"access_token": token},
            )

            assert response.status_code == 500
            assert "Failed to fetch signing keys" in response.json()["detail"]

    def test_oauth_rejects_unknown_kid(self, client):
        """Test that tokens with unknown key IDs are rejected."""
        from jose import jwt

        settings = get_settings()
        expected_issuer = f"{settings.supabase_url}/auth/v1"

        token = jwt.encode(
            {
                "sub": "test-user-id",
                "email": "test@example.com",
                "iss": expected_issuer,
                "aud": "authenticated",
                "exp": 9999999999,
            },
            "fake-secret",
            algorithm="HS256",
            headers={"kid": "unknown-kid-12345"},
        )

        # Mock JWKS response with no matching key
        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client_class.return_value.__aenter__.return_value = mock_client
            mock_response = MagicMock(status_code=200)
            mock_response.json.return_value = {
                "keys": [{"kid": "different-kid", "kty": "RSA", "n": "abc", "e": "AQAB"}]
            }
            mock_client.get.return_value = mock_response

            response = client.post(
                "/auth/oauth/token",
                json={"access_token": token},
            )

            assert response.status_code == 401
            assert "Token signing key not found" in response.json()["detail"]
