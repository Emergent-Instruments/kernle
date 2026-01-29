"""Tests for cached quota checking."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from app.auth import (
    _quota_cache,
    _quota_cache_lock,
    check_quota_cached,
)
from fastapi import HTTPException


@pytest.fixture(autouse=True)
def clear_cache():
    """Clear cache before each test."""
    with _quota_cache_lock:
        _quota_cache.clear()
    yield
    with _quota_cache_lock:
        _quota_cache.clear()


@pytest.mark.asyncio
async def test_cache_miss_queries_db_and_caches():
    """Test that cache miss queries DB and stores result."""
    mock_db = MagicMock()

    with patch("app.database.check_quota", new_callable=AsyncMock) as mock_check:
        mock_check.return_value = (True, {"tier": "free", "daily_requests": 5})

        # First call - cache miss
        allowed, info = await check_quota_cached(mock_db, "key123", "user456", "free")

        assert allowed is True
        assert info["tier"] == "free"
        mock_check.assert_called_once()

        # Verify it's cached
        with _quota_cache_lock:
            cached = _quota_cache.get("key123")
        assert cached is not None
        assert cached[0] is True


@pytest.mark.asyncio
async def test_cache_hit_returns_cached_value():
    """Test that cache hit returns cached value without DB query."""
    mock_db = MagicMock()

    # Pre-populate cache
    with _quota_cache_lock:
        _quota_cache["key123"] = (True, {"tier": "pro", "cached": True})

    with patch("app.database.check_quota", new_callable=AsyncMock) as mock_check:
        allowed, info = await check_quota_cached(mock_db, "key123", "user456", "free")

        assert allowed is True
        assert info["cached"] is True
        # DB should NOT be called
        mock_check.assert_not_called()


@pytest.mark.asyncio
async def test_db_error_with_no_cache_raises_503():
    """Test that DB error with no cache returns 503."""
    mock_db = MagicMock()

    with patch("app.database.check_quota", new_callable=AsyncMock) as mock_check:
        mock_check.side_effect = Exception("Database connection failed")

        with pytest.raises(HTTPException) as exc_info:
            await check_quota_cached(mock_db, "key123", "user456", "free")

        assert exc_info.value.status_code == 503
        assert "temporarily unavailable" in exc_info.value.detail


@pytest.mark.asyncio
async def test_cache_stores_deny_decisions():
    """Test that denied quota decisions are also cached."""
    mock_db = MagicMock()

    with patch("app.database.check_quota", new_callable=AsyncMock) as mock_check:
        mock_check.return_value = (False, {"tier": "free", "exceeded": "daily"})

        # First call - should cache the denial
        allowed, info = await check_quota_cached(mock_db, "key123", "user456", "free")

        assert allowed is False
        assert info["exceeded"] == "daily"

        # Second call - should use cache
        mock_check.reset_mock()
        allowed2, info2 = await check_quota_cached(mock_db, "key123", "user456", "free")

        assert allowed2 is False
        mock_check.assert_not_called()  # Should not query DB again


@pytest.mark.asyncio
async def test_different_keys_cached_separately():
    """Test that different API keys have separate cache entries."""
    mock_db = MagicMock()

    with patch("app.database.check_quota", new_callable=AsyncMock) as mock_check:
        # First key - allowed
        mock_check.return_value = (True, {"key": "1"})
        await check_quota_cached(mock_db, "key1", "user1", "free")

        # Second key - denied
        mock_check.return_value = (False, {"key": "2"})
        await check_quota_cached(mock_db, "key2", "user2", "free")

        # Verify separate cache entries
        with _quota_cache_lock:
            cached1 = _quota_cache.get("key1")
            cached2 = _quota_cache.get("key2")

        assert cached1[0] is True  # key1 allowed
        assert cached2[0] is False  # key2 denied
