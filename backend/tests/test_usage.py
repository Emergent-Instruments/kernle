"""Test usage tracking and tier limits."""

from app.auth import AuthContext
from app.database import TIER_LIMITS
from app.models import UsageLimits, UsageResponse, UsageStats


class TestTierLimits:
    """Test tier limit configurations."""

    def test_free_tier_limits(self):
        """Test free tier has expected limits."""
        limits = TIER_LIMITS["free"]
        assert limits["daily"] == 100
        assert limits["monthly"] == 1000

    def test_unlimited_tier_no_limits(self):
        """Test unlimited tier has no limits."""
        limits = TIER_LIMITS["unlimited"]
        assert limits["daily"] is None
        assert limits["monthly"] is None

    def test_paid_tier_higher_limits(self):
        """Test paid tier has higher limits than free."""
        free_limits = TIER_LIMITS["free"]
        paid_limits = TIER_LIMITS["paid"]
        assert paid_limits["daily"] > free_limits["daily"]
        assert paid_limits["monthly"] > free_limits["monthly"]


class TestAuthContextTier:
    """Test AuthContext includes tier."""

    def test_auth_context_default_tier(self):
        """Test AuthContext defaults to free tier."""
        ctx = AuthContext(agent_id="test", user_id="usr_123")
        assert ctx.tier == "free"

    def test_auth_context_custom_tier(self):
        """Test AuthContext accepts custom tier."""
        ctx = AuthContext(agent_id="test", user_id="usr_123", tier="unlimited")
        assert ctx.tier == "unlimited"

    def test_auth_context_api_key_id(self):
        """Test AuthContext includes api_key_id."""
        ctx = AuthContext(agent_id="test", user_id="usr_123", tier="free", api_key_id="key_123")
        assert ctx.api_key_id == "key_123"


class TestUsageModels:
    """Test usage-related Pydantic models."""

    def test_usage_limits_unlimited(self):
        """Test UsageLimits with no limits."""
        limits = UsageLimits(daily_limit=None, monthly_limit=None)
        assert limits.daily_limit is None
        assert limits.monthly_limit is None

    def test_usage_limits_with_values(self):
        """Test UsageLimits with values."""
        limits = UsageLimits(daily_limit=100, monthly_limit=1000)
        assert limits.daily_limit == 100
        assert limits.monthly_limit == 1000

    def test_usage_stats_defaults(self):
        """Test UsageStats default values."""
        stats = UsageStats()
        assert stats.daily_requests == 0
        assert stats.monthly_requests == 0
        assert stats.daily_reset_at is None
        assert stats.monthly_reset_at is None

    def test_usage_response_model(self):
        """Test UsageResponse model."""
        response = UsageResponse(
            tier="free",
            limits=UsageLimits(daily_limit=100, monthly_limit=1000),
            usage=UsageStats(daily_requests=50, monthly_requests=500),
            daily_remaining=50,
            monthly_remaining=500,
        )
        assert response.tier == "free"
        assert response.limits.daily_limit == 100
        assert response.usage.daily_requests == 50
        assert response.daily_remaining == 50

    def test_usage_response_unlimited(self):
        """Test UsageResponse for unlimited tier."""
        response = UsageResponse(
            tier="unlimited",
            limits=UsageLimits(daily_limit=None, monthly_limit=None),
            usage=UsageStats(daily_requests=0, monthly_requests=0),
            daily_remaining=None,
            monthly_remaining=None,
        )
        assert response.tier == "unlimited"
        assert response.daily_remaining is None


class TestUsageEndpoints:
    """Test usage API endpoints."""

    def test_usage_requires_auth(self, client):
        """Test /auth/usage requires authentication."""
        response = client.get("/auth/usage")
        assert response.status_code == 401

    def test_usage_with_auth(self, client, auth_headers):
        """Test /auth/usage with valid auth returns usage data."""
        from unittest.mock import AsyncMock, patch

        mock_user = {
            "user_id": "usr_TEST_ONLY_000000",
            "email": "test@example.com",
            "tier": "free",
        }
        mock_usage = {
            "daily_requests": 25,
            "monthly_requests": 150,
            "daily_reset_at": "2024-01-02T00:00:00Z",
            "monthly_reset_at": "2024-02-01T00:00:00Z",
        }

        with (
            patch("app.routes.auth.get_user", new_callable=AsyncMock) as mock_get_user,
            patch("app.routes.auth.get_usage_for_user", new_callable=AsyncMock) as mock_get_usage,
        ):
            mock_get_user.return_value = mock_user
            mock_get_usage.return_value = mock_usage

            response = client.get("/auth/usage", headers=auth_headers)

            assert response.status_code == 200, f"Expected 200, got {response.status_code}"
            data = response.json()

            # Verify response structure and values
            assert data["tier"] == "free"
            assert "limits" in data
            assert data["limits"]["daily_limit"] == 100  # free tier limit
            assert data["limits"]["monthly_limit"] == 1000  # free tier limit
            assert "usage" in data
            assert data["usage"]["daily_requests"] == 25
            assert data["usage"]["monthly_requests"] == 150
            # Verify remaining calculations
            assert data["daily_remaining"] == 75  # 100 - 25
            assert data["monthly_remaining"] == 850  # 1000 - 150

            # Verify mocks were called
            mock_get_user.assert_called_once()
            mock_get_usage.assert_called_once()
