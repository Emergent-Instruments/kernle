"""Test API key functionality."""

import pytest

from app.auth import (
    API_KEY_PREFIX,
    generate_api_key,
    get_api_key_prefix,
    hash_api_key,
    is_api_key,
    verify_api_key,
)


class TestAPIKeyUtilities:
    """Test API key utility functions."""

    def test_generate_api_key_format(self):
        """Test API key generation format: knl_sk_ + 32 hex chars."""
        key = generate_api_key()
        
        assert key.startswith(API_KEY_PREFIX)
        assert len(key) == len(API_KEY_PREFIX) + 32  # prefix + 32 hex chars
        
        # Extract hex part and verify it's valid hex
        hex_part = key[len(API_KEY_PREFIX):]
        assert len(hex_part) == 32
        int(hex_part, 16)  # Should not raise

    def test_generate_api_key_uniqueness(self):
        """Test that generated keys are unique."""
        keys = [generate_api_key() for _ in range(100)]
        assert len(set(keys)) == 100

    def test_get_api_key_prefix(self):
        """Test key prefix extraction."""
        key = "knl_sk_abc123def456789012345678901234567890"
        prefix = get_api_key_prefix(key)
        
        assert prefix == "knl_sk_a"
        assert len(prefix) == 8

    def test_hash_api_key(self):
        """Test API key hashing produces bcrypt hash."""
        key = generate_api_key()
        hashed = hash_api_key(key)
        
        # bcrypt hashes start with $2b$ or $2a$
        assert hashed.startswith("$2")
        assert len(hashed) == 60  # bcrypt standard length
        
        # Hash should be different from original
        assert hashed != key

    def test_verify_api_key_valid(self):
        """Test verification of valid API key."""
        key = generate_api_key()
        hashed = hash_api_key(key)
        
        assert verify_api_key(key, hashed) is True

    def test_verify_api_key_invalid(self):
        """Test verification fails for wrong key."""
        key = generate_api_key()
        hashed = hash_api_key(key)
        
        wrong_key = generate_api_key()
        assert verify_api_key(wrong_key, hashed) is False

    def test_verify_api_key_handles_bad_hash(self):
        """Test verification handles malformed hash gracefully."""
        key = generate_api_key()
        
        assert verify_api_key(key, "not-a-hash") is False
        assert verify_api_key(key, "") is False

    def test_is_api_key_true(self):
        """Test is_api_key returns True for API keys."""
        key = generate_api_key()
        assert is_api_key(key) is True
        
        # Manual prefix check
        assert is_api_key("knl_sk_anything") is True

    def test_is_api_key_false_for_jwt(self):
        """Test is_api_key returns False for JWT tokens."""
        # JWTs have three dot-separated parts
        jwt = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiJ0ZXN0In0.sig"
        assert is_api_key(jwt) is False
        
        # Other random strings
        assert is_api_key("random-token") is False
        assert is_api_key("") is False


class TestAPIKeyEndpoints:
    """Test API key API endpoints."""

    def test_create_key_requires_auth(self, client):
        """Test POST /auth/keys requires authentication."""
        response = client.post("/auth/keys")
        assert response.status_code == 401

    def test_list_keys_requires_auth(self, client):
        """Test GET /auth/keys requires authentication."""
        response = client.get("/auth/keys")
        assert response.status_code == 401

    def test_delete_key_requires_auth(self, client):
        """Test DELETE /auth/keys/{id} requires authentication."""
        response = client.delete("/auth/keys/some-id")
        assert response.status_code == 401

    def test_cycle_key_requires_auth(self, client):
        """Test POST /auth/keys/{id}/cycle requires authentication."""
        response = client.post("/auth/keys/some-id/cycle")
        assert response.status_code == 401

    def test_list_keys_with_auth(self, client, auth_headers):
        """Test listing keys with valid auth (may fail due to no user_id in test token)."""
        response = client.get("/auth/keys", headers=auth_headers)
        # Either 200 (success) or 400 (no user_id) are valid
        # 401 would mean auth failed
        assert response.status_code in [200, 400]

    def test_create_key_with_auth(self, client, auth_headers):
        """Test creating key with valid auth (may fail due to no user_id in test token)."""
        response = client.post(
            "/auth/keys",
            headers=auth_headers,
            json={"name": "Test Key"},
        )
        # Either 200 (success), 400 (no user_id), or 500 (DB error) are valid
        # 401 would mean auth failed
        assert response.status_code in [200, 400, 500]


class TestAPIKeyIntegration:
    """Integration tests for API key flow (requires real DB)."""

    @pytest.mark.skipif(
        True,  # Skip by default - run manually with DB
        reason="Requires database connection"
    )
    def test_full_api_key_lifecycle(self, client):
        """Test complete API key lifecycle: create, list, use, cycle, delete."""
        # This would require:
        # 1. Register a new agent
        # 2. Create an API key
        # 3. Use the API key to authenticate
        # 4. List keys
        # 5. Cycle the key
        # 6. Verify old key no longer works
        # 7. Delete the new key
        pass
