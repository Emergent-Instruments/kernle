"""Tests for atomic quota checking with caching."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from app.auth import (
    _quota_cache,
    _quota_cache_lock,
    check_and_increment_quota_cached,
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
async def test_allow_always_queries_db():
    """Test that allowed requests always query DB (for atomic increment)."""
    mock_db = MagicMock()

    with patch("app.database.check_and_increment_quota", new_callable=AsyncMock) as mock_check:
        mock_check.return_value = (True, {"tier": "free", "daily_requests": 5})

        # First call
        allowed, info = await check_and_increment_quota_cached(mock_db, "key123", "user456", "free")
        assert allowed is True
        assert info["tier"] == "free"
        mock_check.assert_called_once()

        # Second call - should still query DB (no caching for allows)
        mock_check.reset_mock()
        allowed2, _ = await check_and_increment_quota_cached(mock_db, "key123", "user456", "free")
        assert allowed2 is True
        mock_check.assert_called_once()  # Called again


@pytest.mark.asyncio
async def test_denial_is_cached():
    """Test that denied requests are cached."""
    mock_db = MagicMock()

    with patch("app.database.check_and_increment_quota", new_callable=AsyncMock) as mock_check:
        mock_check.return_value = (False, {"tier": "free", "exceeded": "daily"})

        # First call - should query DB and cache denial
        allowed, info = await check_and_increment_quota_cached(mock_db, "key123", "user456", "free")
        assert allowed is False
        assert info["exceeded"] == "daily"
        mock_check.assert_called_once()

        # Second call - should use cached denial
        mock_check.reset_mock()
        allowed2, _ = await check_and_increment_quota_cached(mock_db, "key123", "user456", "free")
        assert allowed2 is False
        mock_check.assert_not_called()  # Should use cache


@pytest.mark.asyncio
async def test_cached_denial_returns_info():
    """Test that cached denial returns quota info."""
    mock_db = MagicMock()

    # Pre-populate cache with denial
    denial_info = {"tier": "free", "exceeded": "daily", "daily_reset_at": "2024-01-01T00:00:00Z"}
    with _quota_cache_lock:
        _quota_cache["deny:key123"] = denial_info

    with patch("app.database.check_and_increment_quota", new_callable=AsyncMock) as mock_check:
        allowed, info = await check_and_increment_quota_cached(mock_db, "key123", "user456", "free")

        assert allowed is False
        assert info["exceeded"] == "daily"
        mock_check.assert_not_called()


@pytest.mark.asyncio
async def test_db_error_raises_503():
    """Test that DB error returns 503 (fail closed)."""
    mock_db = MagicMock()

    with patch("app.database.check_and_increment_quota", new_callable=AsyncMock) as mock_check:
        mock_check.side_effect = Exception("Database connection failed")

        with pytest.raises(HTTPException) as exc_info:
            await check_and_increment_quota_cached(mock_db, "key123", "user456", "free")

        assert exc_info.value.status_code == 503
        assert "temporarily unavailable" in exc_info.value.detail


@pytest.mark.asyncio
async def test_different_keys_cached_separately():
    """Test that different API keys have separate cache entries for denials."""
    mock_db = MagicMock()

    with patch("app.database.check_and_increment_quota", new_callable=AsyncMock) as mock_check:
        # First key - denied
        mock_check.return_value = (False, {"key": "1", "exceeded": "daily"})
        await check_and_increment_quota_cached(mock_db, "key1", "user1", "free")

        # Second key - also denied
        mock_check.return_value = (False, {"key": "2", "exceeded": "monthly"})
        await check_and_increment_quota_cached(mock_db, "key2", "user2", "free")

        # Verify separate cache entries
        with _quota_cache_lock:
            cached1 = _quota_cache.get("deny:key1")
            cached2 = _quota_cache.get("deny:key2")

        assert cached1["key"] == "1"
        assert cached2["key"] == "2"


@pytest.mark.asyncio
async def test_atomic_increment_called_on_allow():
    """Test that the atomic check-and-increment is called for allowed requests."""
    mock_db = MagicMock()

    with patch("app.database.check_and_increment_quota", new_callable=AsyncMock) as mock_check:
        mock_check.return_value = (True, {"tier": "free", "daily_requests": 1})

        await check_and_increment_quota_cached(mock_db, "key123", "user456", "free")

        # Verify the atomic function was called with correct args
        mock_check.assert_called_once_with(mock_db, "key123", "user456", "free")
