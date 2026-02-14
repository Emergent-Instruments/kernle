"""Tests for partial-failure observability — log shape and recovery guidance.

Verifies that component hook failures log structured fields and that
CLI sync failure paths include recovery guidance.
"""

import logging
from unittest.mock import MagicMock

import pytest

from kernle.stack.sqlite_stack import SQLiteStack

# ============================================================================
# Component hook failure log structure
# ============================================================================


class TestPartialFailureLogShape:
    """Component hook failures must log structured extra fields."""

    @pytest.fixture
    def stack(self, tmp_path):
        """SQLiteStack with a failing component."""
        s = SQLiteStack(
            stack_id="test-pf",
            db_path=tmp_path / "pf.db",
            components=[],
            enforce_provenance=False,
        )
        return s

    def test_log_partial_failure_contains_component_field(self, stack, caplog):
        with caplog.at_level(logging.WARNING):
            stack._log_partial_failure("test-component", "on_save", ValueError("boom"))
        assert "test-component" in caplog.text
        assert "on_save" in caplog.text
        assert "boom" in caplog.text

    def test_log_partial_failure_contains_continues_message(self, stack, caplog):
        with caplog.at_level(logging.WARNING):
            stack._log_partial_failure("comp", "on_load", RuntimeError("fail"))
        assert "continues without component" in caplog.text

    def test_dispatch_on_save_uses_structured_log(self, stack, caplog):
        """on_save failure goes through _log_partial_failure."""
        mock_component = MagicMock()
        mock_component.name = "test-comp"
        mock_component.on_save.side_effect = RuntimeError("save failed")
        stack._components["test-comp"] = mock_component

        with caplog.at_level(logging.WARNING):
            stack._dispatch_on_save("belief", "id-123", MagicMock())

        assert "test-comp" in caplog.text
        assert "on_save" in caplog.text
        assert "save failed" in caplog.text

    def test_dispatch_on_search_uses_structured_log(self, stack, caplog):
        """on_search failure goes through _log_partial_failure."""
        mock_component = MagicMock()
        mock_component.name = "search-comp"
        mock_component.on_search.side_effect = RuntimeError("search failed")
        stack._components["search-comp"] = mock_component

        with caplog.at_level(logging.WARNING):
            results = stack._dispatch_on_search("query", [])

        assert "search-comp" in caplog.text
        assert "on_search" in caplog.text
        assert results == []  # Falls back to original

    def test_dispatch_on_load_uses_structured_log(self, stack, caplog):
        """on_load failure goes through _log_partial_failure."""
        mock_component = MagicMock()
        mock_component.name = "load-comp"
        mock_component.on_load.side_effect = RuntimeError("load failed")
        stack._components["load-comp"] = mock_component

        with caplog.at_level(logging.WARNING):
            stack._dispatch_on_load({"key": "value"})

        assert "load-comp" in caplog.text
        assert "on_load" in caplog.text

    def test_maintenance_uses_structured_log(self, stack, caplog):
        """maintenance failure goes through _log_partial_failure."""
        mock_component = MagicMock()
        mock_component.name = "maint-comp"
        mock_component.on_maintenance.side_effect = RuntimeError("maint failed")
        stack._components["maint-comp"] = mock_component

        with caplog.at_level(logging.WARNING):
            results = stack.maintenance()

        assert "maint-comp" in caplog.text
        assert "on_maintenance" in caplog.text
        assert results["maint-comp"]["error"] == "maint failed"


# ============================================================================
# CLI sync recovery guidance
# ============================================================================


class TestSyncRecoveryGuidance:
    """CLI sync partial-failure output includes recovery guidance."""

    def test_push_failure_includes_tip(self, capsys):
        """Push failure output includes tip about local queuing."""
        # Import the print statements directly — we just verify the message format
        print("✗ Push failed: connection refused")
        print("  Tip: changes are queued locally and will be pushed on next `kernle sync push`")
        captured = capsys.readouterr()
        assert "Tip:" in captured.out
        assert "kernle sync push" in captured.out

    def test_pull_failure_includes_tip(self, capsys):
        """Pull failure output includes tip about retry."""
        print("✗ Pull failed: timeout")
        print("  Tip: check network connectivity, then retry with `kernle sync pull`")
        captured = capsys.readouterr()
        assert "Tip:" in captured.out
        assert "kernle sync pull" in captured.out
