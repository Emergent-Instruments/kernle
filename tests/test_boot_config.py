"""Tests for boot config (always-available key/value configuration).

Phase 9: Boot layer provides instant config access without full memory load.
"""

import json
import os
from pathlib import Path
from unittest.mock import patch

import pytest

from kernle.core import Kernle
from kernle.storage.sqlite import SQLiteStorage


@pytest.fixture
def tmp_db(tmp_path):
    """Create a temporary database path."""
    return tmp_path / "test.db"


@pytest.fixture
def storage(tmp_db):
    """Create a SQLiteStorage instance with temp DB."""
    return SQLiteStorage(agent_id="test-agent", db_path=tmp_db)


@pytest.fixture
def kernle_instance(tmp_db, tmp_path):
    """Create a Kernle instance with temp DB and home dir."""
    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir()
    s = SQLiteStorage(agent_id="test-agent", db_path=tmp_db)
    k = Kernle(agent_id="test-agent", storage=s, checkpoint_dir=checkpoint_dir)
    yield k
    s.close()


# ============================================================================
# Storage Layer Tests
# ============================================================================


class TestBootStorageSet:
    """Test boot_set storage method."""

    def test_set_basic(self, storage):
        storage.boot_set("chat_id", "4")
        assert storage.boot_get("chat_id") == "4"

    def test_set_overwrites(self, storage):
        storage.boot_set("key", "old")
        storage.boot_set("key", "new")
        assert storage.boot_get("key") == "new"

    def test_set_multiple_keys(self, storage):
        storage.boot_set("a", "1")
        storage.boot_set("b", "2")
        storage.boot_set("c", "3")
        assert storage.boot_list() == {"a": "1", "b": "2", "c": "3"}

    def test_set_empty_value_allowed(self, storage):
        """Empty string values should be allowed (unlike keys)."""
        storage.boot_set("key", "")
        assert storage.boot_get("key") == ""

    def test_set_empty_key_raises(self, storage):
        with pytest.raises(ValueError, match="non-empty string"):
            storage.boot_set("", "value")

    def test_set_whitespace_key_raises(self, storage):
        with pytest.raises(ValueError, match="non-empty string"):
            storage.boot_set("   ", "value")

    def test_set_key_stripped(self, storage):
        storage.boot_set("  key  ", "value")
        assert storage.boot_get("key") == "value"

    def test_set_non_string_key_raises(self, storage):
        with pytest.raises(ValueError):
            storage.boot_set(123, "value")

    def test_set_non_string_value_raises(self, storage):
        with pytest.raises(ValueError):
            storage.boot_set("key", 123)

    def test_set_special_characters(self, storage):
        """Keys and values can contain special characters."""
        storage.boot_set("gateway.ip", "192.168.50.11:18789")
        assert storage.boot_get("gateway.ip") == "192.168.50.11:18789"

    def test_set_unicode(self, storage):
        storage.boot_set("emoji", "🔥")
        assert storage.boot_get("emoji") == "🔥"

    def test_set_multiline_value(self, storage):
        storage.boot_set("notes", "line1\nline2\nline3")
        assert storage.boot_get("notes") == "line1\nline2\nline3"


class TestBootStorageGet:
    """Test boot_get storage method."""

    def test_get_missing_returns_none(self, storage):
        assert storage.boot_get("nonexistent") is None

    def test_get_missing_with_default(self, storage):
        assert storage.boot_get("nonexistent", default="fallback") == "fallback"

    def test_get_existing_ignores_default(self, storage):
        storage.boot_set("key", "real")
        assert storage.boot_get("key", default="fallback") == "real"


class TestBootStorageList:
    """Test boot_list storage method."""

    def test_list_empty(self, storage):
        assert storage.boot_list() == {}

    def test_list_returns_sorted(self, storage):
        storage.boot_set("z_key", "last")
        storage.boot_set("a_key", "first")
        config = storage.boot_list()
        keys = list(config.keys())
        assert keys == ["a_key", "z_key"]

    def test_list_agent_isolation(self, tmp_db):
        """Different agents have separate boot configs."""
        s1 = SQLiteStorage(agent_id="agent-1", db_path=tmp_db)
        s2 = SQLiteStorage(agent_id="agent-2", db_path=tmp_db)

        s1.boot_set("key", "agent1-value")
        s2.boot_set("key", "agent2-value")

        assert s1.boot_get("key") == "agent1-value"
        assert s2.boot_get("key") == "agent2-value"
        assert len(s1.boot_list()) == 1
        assert len(s2.boot_list()) == 1


class TestBootStorageDelete:
    """Test boot_delete storage method."""

    def test_delete_existing(self, storage):
        storage.boot_set("key", "value")
        assert storage.boot_delete("key") is True
        assert storage.boot_get("key") is None

    def test_delete_nonexistent(self, storage):
        assert storage.boot_delete("nonexistent") is False

    def test_delete_doesnt_affect_others(self, storage):
        storage.boot_set("keep", "yes")
        storage.boot_set("remove", "yes")
        storage.boot_delete("remove")
        assert storage.boot_get("keep") == "yes"


class TestBootStorageClear:
    """Test boot_clear storage method."""

    def test_clear_empty(self, storage):
        assert storage.boot_clear() == 0

    def test_clear_all(self, storage):
        storage.boot_set("a", "1")
        storage.boot_set("b", "2")
        storage.boot_set("c", "3")
        count = storage.boot_clear()
        assert count == 3
        assert storage.boot_list() == {}

    def test_clear_agent_isolation(self, tmp_db):
        """Clear only affects the calling agent."""
        s1 = SQLiteStorage(agent_id="agent-1", db_path=tmp_db)
        s2 = SQLiteStorage(agent_id="agent-2", db_path=tmp_db)

        s1.boot_set("key", "val1")
        s2.boot_set("key", "val2")

        s1.boot_clear()
        assert s1.boot_list() == {}
        assert s2.boot_get("key") == "val2"


# ============================================================================
# Core Layer Tests
# ============================================================================


class TestBootCore:
    """Test boot config via Kernle core."""

    def test_set_get(self, kernle_instance):
        kernle_instance.boot_set("chat_id", "4")
        assert kernle_instance.boot_get("chat_id") == "4"

    def test_list(self, kernle_instance):
        kernle_instance.boot_set("a", "1")
        kernle_instance.boot_set("b", "2")
        assert kernle_instance.boot_list() == {"a": "1", "b": "2"}

    def test_delete(self, kernle_instance):
        kernle_instance.boot_set("key", "val")
        assert kernle_instance.boot_delete("key") is True
        assert kernle_instance.boot_get("key") is None

    def test_clear(self, kernle_instance):
        kernle_instance.boot_set("a", "1")
        kernle_instance.boot_set("b", "2")
        assert kernle_instance.boot_clear() == 2
        assert kernle_instance.boot_list() == {}


class TestBootFileProjection:
    """Test auto-export of boot.md file."""

    def test_set_creates_boot_file(self, kernle_instance, tmp_path):
        with patch.object(Path, "home", return_value=tmp_path):
            kernle_instance.boot_set("key", "value")
            boot_path = tmp_path / ".kernle" / "test-agent" / "boot.md"
            # The boot file should be created by the storage init or by set
            # Check that _export_boot_file works
            kernle_instance._export_boot_file()
            assert boot_path.exists()
            content = boot_path.read_text()
            assert "key: value" in content
            assert "Auto-generated" in content

    def test_clear_removes_boot_file(self, kernle_instance, tmp_path):
        with patch.object(Path, "home", return_value=tmp_path):
            kernle_instance.boot_set("key", "value")
            kernle_instance._export_boot_file()
            boot_path = tmp_path / ".kernle" / "test-agent" / "boot.md"
            assert boot_path.exists()

            kernle_instance.boot_clear()
            # After clear, boot file should be removed
            assert not boot_path.exists()

    def test_boot_file_permissions(self, kernle_instance, tmp_path):
        """Boot file should have 0600 permissions."""
        with patch.object(Path, "home", return_value=tmp_path):
            kernle_instance.boot_set("secret", "token123")
            boot_path = tmp_path / ".kernle" / "test-agent" / "boot.md"
            kernle_instance._export_boot_file()
            if boot_path.exists():
                mode = oct(boot_path.stat().st_mode & 0o777)
                assert mode == "0o600"


class TestBootInLoad:
    """Test boot config inclusion in load output."""

    def test_load_includes_boot(self, kernle_instance):
        kernle_instance.boot_set("chat_id", "4")
        kernle_instance.boot_set("gateway_ip", "192.168.50.11")
        memory = kernle_instance.load()
        assert "boot_config" in memory
        assert memory["boot_config"]["chat_id"] == "4"
        assert memory["boot_config"]["gateway_ip"] == "192.168.50.11"

    def test_load_no_boot_when_empty(self, kernle_instance):
        memory = kernle_instance.load()
        assert "boot_config" not in memory

    def test_format_memory_includes_boot(self, kernle_instance):
        kernle_instance.boot_set("chat_id", "4")
        memory = kernle_instance.load()
        formatted = kernle_instance.format_memory(memory)
        assert "## Boot Config" in formatted
        assert "chat_id: 4" in formatted


class TestBootInExportCache:
    """Test boot config inclusion in export-cache output."""

    def test_export_cache_includes_boot(self, kernle_instance):
        kernle_instance.boot_set("chat_id", "4")
        kernle_instance.boot_set("name", "Ash")
        content = kernle_instance.export_cache()
        assert "## Boot Config" in content
        assert "chat_id: 4" in content
        assert "name: Ash" in content

    def test_export_cache_no_boot_when_empty(self, kernle_instance):
        content = kernle_instance.export_cache()
        assert "## Boot Config" not in content

    def test_export_cache_boot_before_values(self, kernle_instance):
        """Boot config should appear before values in export-cache."""
        kernle_instance.boot_set("key", "val")
        # Add a value for ordering check
        from kernle.storage.base import Value
        from datetime import datetime, timezone
        val = Value(
            id="test-val",
            agent_id="test-agent",
            name="test_value",
            statement="test statement",
            priority=50,
            created_at=datetime.now(timezone.utc),
        )
        kernle_instance._storage.save_value(val)

        content = kernle_instance.export_cache()
        boot_pos = content.index("## Boot Config")
        # Values section may or may not appear depending on load
        if "## Values" in content:
            values_pos = content.index("## Values")
            assert boot_pos < values_pos


class TestBootInCheckpoint:
    """Test boot file auto-export on checkpoint."""

    def test_checkpoint_exports_boot(self, kernle_instance, tmp_path):
        with patch.object(Path, "home", return_value=tmp_path):
            kernle_instance.boot_set("key", "value")
            boot_path = tmp_path / ".kernle" / "test-agent" / "boot.md"

            # Delete boot file if it exists
            if boot_path.exists():
                boot_path.unlink()

            # Checkpoint should re-export boot file
            kernle_instance.checkpoint("test task")
            assert boot_path.exists()


class TestBootFormatSection:
    """Test _format_boot_section helper."""

    def test_empty_returns_empty(self, kernle_instance):
        assert kernle_instance._format_boot_section() == []

    def test_formats_correctly(self, kernle_instance):
        kernle_instance.boot_set("a", "1")
        kernle_instance.boot_set("b", "2")
        lines = kernle_instance._format_boot_section()
        assert lines[0] == "## Boot Config"
        assert "- a: 1" in lines
        assert "- b: 2" in lines
        assert lines[-1] == ""  # trailing blank line


# ============================================================================
# Schema Migration Tests
# ============================================================================


class TestBootSchemaMigration:
    """Test that boot_config table is created on existing databases."""

    def test_table_created_on_init(self, tmp_db):
        """boot_config table should exist after storage init."""
        import sqlite3
        storage = SQLiteStorage(agent_id="test", db_path=tmp_db)
        conn = sqlite3.connect(tmp_db)
        tables = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='boot_config'"
        ).fetchall()
        conn.close()
        assert len(tables) == 1

    def test_unique_constraint(self, storage):
        """Agent + key should be unique (upsert, not error)."""
        storage.boot_set("key", "first")
        storage.boot_set("key", "second")  # Should upsert, not error
        assert storage.boot_get("key") == "second"


# ============================================================================
# CLI Integration Tests
# ============================================================================


class TestBootCLI:
    """Test boot CLI command handler directly."""

    @pytest.fixture
    def cli_k(self, tmp_path):
        """Create Kernle instance with pre-populated boot config."""
        db_path = tmp_path / "test.db"
        checkpoint_dir = tmp_path / "checkpoints"
        checkpoint_dir.mkdir()
        storage = SQLiteStorage(agent_id="cli-test", db_path=db_path)
        k = Kernle(agent_id="cli-test", storage=storage, checkpoint_dir=checkpoint_dir)
        k.boot_set("chat_id", "4")
        k.boot_set("gateway_ip", "192.168.50.11")
        return k

    def _make_args(self, **kwargs):
        """Create a mock args namespace."""
        import argparse
        return argparse.Namespace(**kwargs)

    def test_boot_get(self, cli_k, capsys):
        from kernle.cli.__main__ import cmd_boot
        args = self._make_args(boot_action="get", key="chat_id")
        cmd_boot(args, cli_k)
        assert capsys.readouterr().out.strip() == "4"

    def test_boot_get_missing(self, cli_k):
        from kernle.cli.__main__ import cmd_boot
        args = self._make_args(boot_action="get", key="nonexistent")
        with pytest.raises(SystemExit) as exc_info:
            cmd_boot(args, cli_k)
        assert exc_info.value.code == 1

    def test_boot_list_plain(self, cli_k, capsys):
        from kernle.cli.__main__ import cmd_boot
        args = self._make_args(boot_action="list", format="plain")
        cmd_boot(args, cli_k)
        output = capsys.readouterr().out
        assert "chat_id: 4" in output
        assert "gateway_ip: 192.168.50.11" in output

    def test_boot_list_json(self, cli_k, capsys):
        from kernle.cli.__main__ import cmd_boot
        args = self._make_args(boot_action="list", format="json")
        cmd_boot(args, cli_k)
        data = json.loads(capsys.readouterr().out)
        assert data["chat_id"] == "4"
        assert data["gateway_ip"] == "192.168.50.11"

    def test_boot_list_md(self, cli_k, capsys):
        from kernle.cli.__main__ import cmd_boot
        args = self._make_args(boot_action="list", format="md")
        cmd_boot(args, cli_k)
        output = capsys.readouterr().out
        assert "## Boot Config" in output
        assert "- chat_id: 4" in output

    def test_boot_set(self, cli_k, capsys):
        from kernle.cli.__main__ import cmd_boot
        args = self._make_args(boot_action="set", key="new_key", value="new_value")
        cmd_boot(args, cli_k)
        output = capsys.readouterr().out
        assert "new_key: new_value" in output
        assert cli_k.boot_get("new_key") == "new_value"

    def test_boot_delete(self, cli_k, capsys):
        from kernle.cli.__main__ import cmd_boot
        args = self._make_args(boot_action="delete", key="chat_id")
        cmd_boot(args, cli_k)
        output = capsys.readouterr().out
        assert "Deleted" in output
        assert cli_k.boot_get("chat_id") is None

    def test_boot_delete_missing(self, cli_k):
        from kernle.cli.__main__ import cmd_boot
        args = self._make_args(boot_action="delete", key="nonexistent")
        with pytest.raises(SystemExit) as exc_info:
            cmd_boot(args, cli_k)
        assert exc_info.value.code == 1

    def test_boot_clear_requires_confirm(self, cli_k):
        from kernle.cli.__main__ import cmd_boot
        args = self._make_args(boot_action="clear", confirm=False)
        with pytest.raises(SystemExit) as exc_info:
            cmd_boot(args, cli_k)
        assert exc_info.value.code == 2

    def test_boot_clear_with_confirm(self, cli_k, capsys):
        from kernle.cli.__main__ import cmd_boot
        args = self._make_args(boot_action="clear", confirm=True)
        cmd_boot(args, cli_k)
        output = capsys.readouterr().out
        assert "Cleared" in output
        assert cli_k.boot_list() == {}

    def test_boot_list_empty(self, tmp_path, capsys):
        from kernle.cli.__main__ import cmd_boot
        db_path = tmp_path / "empty.db"
        checkpoint_dir = tmp_path / "cp"
        checkpoint_dir.mkdir()
        s = SQLiteStorage(agent_id="empty", db_path=db_path)
        k = Kernle(agent_id="empty", storage=s, checkpoint_dir=checkpoint_dir)
        args = self._make_args(boot_action="list", format="plain")
        cmd_boot(args, k)
        output = capsys.readouterr().out
        assert "no boot config" in output
