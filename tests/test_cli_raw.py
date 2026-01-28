"""Tests for CLI raw command module."""

import pytest
from unittest.mock import MagicMock, patch
from argparse import Namespace

from kernle.cli.commands.raw import cmd_raw, resolve_raw_id


class TestResolveRawId:
    """Test raw ID resolution."""

    def test_exact_match(self):
        """Exact ID should resolve directly."""
        k = MagicMock()
        k.get_raw.return_value = {"id": "abc123", "content": "test"}
        
        result = resolve_raw_id(k, "abc123")
        assert result == "abc123"
        k.get_raw.assert_called_once_with("abc123")

    def test_prefix_match_single(self):
        """Single prefix match should resolve to full ID."""
        k = MagicMock()
        k.get_raw.return_value = None  # No exact match
        k.list_raw.return_value = [
            {"id": "abc123456789", "content": "test"},
            {"id": "xyz987654321", "content": "other"},
        ]
        
        result = resolve_raw_id(k, "abc")
        assert result == "abc123456789"

    def test_prefix_match_ambiguous(self):
        """Multiple prefix matches should raise error."""
        k = MagicMock()
        k.get_raw.return_value = None
        k.list_raw.return_value = [
            {"id": "abc123456789", "content": "test1"},
            {"id": "abc987654321", "content": "test2"},
        ]
        
        with pytest.raises(ValueError, match="Ambiguous ID"):
            resolve_raw_id(k, "abc")

    def test_no_match(self):
        """No matches should raise error."""
        k = MagicMock()
        k.get_raw.return_value = None
        k.list_raw.return_value = []
        
        with pytest.raises(ValueError, match="not found"):
            resolve_raw_id(k, "nonexistent")


class TestCmdRawCapture:
    """Test raw capture command."""

    def test_capture_basic(self, capsys):
        """Basic capture should work."""
        k = MagicMock()
        k.raw.return_value = "raw-id-12345678"
        
        args = Namespace(
            raw_action="capture",
            content="test content",
            tags=None,
            source=None,
        )
        
        cmd_raw(args, k)
        
        k.raw.assert_called_once()
        captured = capsys.readouterr()
        assert "✓ Raw entry captured" in captured.out

    def test_capture_with_tags(self, capsys):
        """Capture with tags should pass them through."""
        k = MagicMock()
        k.raw.return_value = "raw-id-12345678"
        
        args = Namespace(
            raw_action="capture",
            content="test content",
            tags="tag1,tag2",
            source=None,
        )
        
        cmd_raw(args, k)
        
        call_kwargs = k.raw.call_args
        assert "tag1" in call_kwargs[1]["tags"]
        assert "tag2" in call_kwargs[1]["tags"]


class TestCmdRawList:
    """Test raw list command."""

    def test_list_empty(self, capsys):
        """Empty list should show message."""
        k = MagicMock()
        k.list_raw.return_value = []
        
        args = Namespace(
            raw_action="list",
            unprocessed=False,
            processed=False,
            limit=20,
            json=False,
        )
        
        cmd_raw(args, k)
        
        captured = capsys.readouterr()
        assert "No raw entries found" in captured.out

    def test_list_unprocessed_filter(self):
        """Unprocessed filter should be passed."""
        k = MagicMock()
        k.list_raw.return_value = []
        
        args = Namespace(
            raw_action="list",
            unprocessed=True,
            processed=False,
            limit=20,
            json=False,
        )
        
        cmd_raw(args, k)
        
        k.list_raw.assert_called_once_with(processed=False, limit=20)


class TestCmdRawShow:
    """Test raw show command."""

    def test_show_not_found(self, capsys):
        """Show non-existent ID should error."""
        k = MagicMock()
        k.get_raw.return_value = None
        k.list_raw.return_value = []
        
        args = Namespace(
            raw_action="show",
            id="nonexistent",
            json=False,
        )
        
        cmd_raw(args, k)
        
        captured = capsys.readouterr()
        assert "not found" in captured.out


class TestCmdRawClean:
    """Test raw clean command."""

    def test_clean_no_targets(self, capsys):
        """No targets should show success message."""
        k = MagicMock()
        k.list_raw.return_value = []
        
        args = Namespace(
            raw_action="clean",
            age=7,
            junk=False,
            confirm=False,
        )
        
        cmd_raw(args, k)
        
        captured = capsys.readouterr()
        assert "No unprocessed raw entries" in captured.out

    def test_clean_junk_detection(self, capsys):
        """Junk mode should detect short entries."""
        k = MagicMock()
        k.list_raw.return_value = [
            {"id": "abc123", "content": "test", "timestamp": "2026-01-01T00:00:00Z"},
            {"id": "def456", "content": "real content here", "timestamp": "2026-01-01T00:00:00Z"},
        ]
        
        args = Namespace(
            raw_action="clean",
            age=7,
            junk=True,
            confirm=False,
        )
        
        cmd_raw(args, k)
        
        captured = capsys.readouterr()
        # "test" is <10 chars, should be detected as junk
        assert "junk" in captured.out.lower()
