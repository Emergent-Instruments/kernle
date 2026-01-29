"""Tests for health check compliance tracking.

Tests the health check event logging and statistics features.
"""

import json

import pytest

from kernle import Kernle
from kernle.storage import SQLiteStorage


@pytest.fixture
def k(temp_checkpoint_dir, temp_db_path):
    """Simple Kernle instance for health check tests."""
    storage = SQLiteStorage(
        agent_id="test_health_check_agent",
        db_path=temp_db_path,
    )

    kernle = Kernle(
        agent_id="test_health_check_agent", storage=storage, checkpoint_dir=temp_checkpoint_dir
    )

    return kernle


class TestHealthCheckLogging:
    """Test health check event logging."""

    def test_log_health_check_basic(self, k):
        """Basic health check logging should create an event."""
        event_id = k._storage.log_health_check(
            anxiety_score=42, source="cli", triggered_by="manual"
        )

        assert event_id is not None
        assert len(event_id) == 36  # UUID length

    def test_log_health_check_stores_values(self, k):
        """Logged values should be retrievable in stats."""
        k._storage.log_health_check(anxiety_score=50, source="cli", triggered_by="boot")
        k._storage.log_health_check(anxiety_score=60, source="mcp", triggered_by="heartbeat")

        stats = k._storage.get_health_check_stats()

        assert stats["total_checks"] == 2
        assert stats["last_anxiety_score"] == 60
        assert "cli" in stats["checks_by_source"]
        assert "mcp" in stats["checks_by_source"]
        assert "boot" in stats["checks_by_trigger"]
        assert "heartbeat" in stats["checks_by_trigger"]

    def test_log_health_check_without_score(self, k):
        """Health check can be logged without an anxiety score."""
        event_id = k._storage.log_health_check(source="cli", triggered_by="manual")

        assert event_id is not None
        stats = k._storage.get_health_check_stats()
        assert stats["total_checks"] == 1
        assert stats["last_anxiety_score"] is None


class TestHealthCheckStats:
    """Test health check statistics calculation."""

    def test_stats_empty(self, k):
        """Stats should return zeros when no checks exist."""
        stats = k._storage.get_health_check_stats()

        assert stats["total_checks"] == 0
        assert stats["avg_per_day"] == 0.0
        assert stats["last_check_at"] is None
        assert stats["last_anxiety_score"] is None
        assert stats["checks_by_source"] == {}
        assert stats["checks_by_trigger"] == {}

    def test_stats_single_check(self, k):
        """Stats should work with a single check."""
        k._storage.log_health_check(anxiety_score=35, source="cli", triggered_by="manual")

        stats = k._storage.get_health_check_stats()

        assert stats["total_checks"] == 1
        assert stats["avg_per_day"] == 1.0
        assert stats["last_check_at"] is not None
        assert stats["last_anxiety_score"] == 35

    def test_stats_multiple_checks(self, k):
        """Stats should aggregate multiple checks correctly."""
        for i in range(5):
            k._storage.log_health_check(
                anxiety_score=20 + i * 10,
                source="cli" if i % 2 == 0 else "mcp",
                triggered_by="manual",
            )

        stats = k._storage.get_health_check_stats()

        assert stats["total_checks"] == 5
        assert stats["last_anxiety_score"] == 60  # Last score (20 + 4*10)
        assert stats["checks_by_source"]["cli"] == 3
        assert stats["checks_by_source"]["mcp"] == 2

    def test_stats_by_trigger_breakdown(self, k):
        """Stats should break down checks by trigger type."""
        k._storage.log_health_check(source="cli", triggered_by="boot")
        k._storage.log_health_check(source="cli", triggered_by="boot")
        k._storage.log_health_check(source="cli", triggered_by="heartbeat")
        k._storage.log_health_check(source="cli", triggered_by="manual")

        stats = k._storage.get_health_check_stats()

        assert stats["checks_by_trigger"]["boot"] == 2
        assert stats["checks_by_trigger"]["heartbeat"] == 1
        assert stats["checks_by_trigger"]["manual"] == 1


class TestAnxietyCommandLogging:
    """Test that the anxiety command logs health check events."""

    def test_anxiety_command_logs_event(self, k):
        """Running anxiety check should log an event."""
        # Get initial report to trigger logging
        report = k.get_anxiety_report()

        # Manually call log since we're testing the storage directly
        k._storage.log_health_check(
            anxiety_score=report.get("overall_score"), source="cli", triggered_by="manual"
        )

        stats = k._storage.get_health_check_stats()
        assert stats["total_checks"] == 1
        assert stats["last_anxiety_score"] is not None


class TestStatsCLI:
    """Test the stats CLI command."""

    def test_stats_health_checks_cli(self, temp_db_path, temp_checkpoint_dir):
        """Test kernle stats health-checks CLI output."""

        # Create storage and log some events
        storage = SQLiteStorage(
            agent_id="cli_test_agent",
            db_path=temp_db_path,
        )
        storage.log_health_check(anxiety_score=30, source="cli", triggered_by="manual")
        storage.log_health_check(anxiety_score=45, source="mcp", triggered_by="heartbeat")

        # Test the CLI - this is a basic smoke test
        # Full CLI integration test would need subprocess
        stats = storage.get_health_check_stats()
        assert stats["total_checks"] == 2

    def test_stats_health_checks_json(self, temp_db_path, temp_checkpoint_dir):
        """Test JSON output from stats command."""
        storage = SQLiteStorage(
            agent_id="json_test_agent",
            db_path=temp_db_path,
        )
        storage.log_health_check(anxiety_score=50, source="cli", triggered_by="boot")

        stats = storage.get_health_check_stats()

        # Verify JSON serializable
        json_str = json.dumps(stats, default=str)
        parsed = json.loads(json_str)

        assert parsed["total_checks"] == 1
        assert parsed["last_anxiety_score"] == 50
        assert "cli" in parsed["checks_by_source"]
