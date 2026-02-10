"""Additional tests for kernle/cli/commands/sync.py to improve coverage.

Targets uncovered lines: legacy config fallback, httpx not installed,
status display formatting, push orphan handling, push non-dataclass
records, pull delete/conflict paths, full sync push paths.
"""

import argparse
import json
import os
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, patch

import pytest

from kernle import Kernle
from kernle.cli.commands.sync import cmd_sync
from kernle.storage import SQLiteStorage

# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def storage(tmp_path):
    s = SQLiteStorage(stack_id="test-sync-cov", db_path=tmp_path / "sync_cov.db")
    yield s
    s.close()


@pytest.fixture
def k(storage):
    inst = Kernle(stack_id="test-sync-cov", storage=storage, strict=False)
    yield inst


def _args(**kwargs):
    defaults = {
        "command": "sync",
        "sync_action": "status",
        "json": False,
        "limit": 100,
        "full": False,
        "clear": False,
    }
    defaults.update(kwargs)
    return argparse.Namespace(**defaults)


def _make_response(status_code=200, json_data=None, text=""):
    resp = MagicMock()
    resp.status_code = status_code
    resp.json.return_value = json_data or {}
    resp.text = text
    return resp


def _setup_creds(tmp_path, creds=None, config=None, dirname="creds"):
    """Create credentials dir with optional credentials.json and config.json."""
    creds_path = tmp_path / dirname
    creds_path.mkdir()
    if creds:
        (creds_path / "credentials.json").write_text(json.dumps(creds))
    if config:
        (creds_path / "config.json").write_text(json.dumps(config))
    return creds_path


# ============================================================================
# Legacy config.json fallback (lines 53-64)
# ============================================================================


class TestLegacyConfigFallback:
    """Test fallback to config.json when credentials.json is missing."""

    def test_legacy_config_json_fallback(self, k, capsys, tmp_path):
        """Falls back to config.json when no credentials.json exists."""
        config = {"backend_url": "https://legacy.api.com", "auth_token": "legacy-tok"}
        creds_path = _setup_creds(tmp_path, config=config, dirname="legacy_config")

        health_resp = _make_response(200)
        mock_httpx = MagicMock()
        mock_httpx.get.return_value = health_resp

        env = {
            "KERNLE_DATA_DIR": str(creds_path),
            "KERNLE_BACKEND_URL": "",
            "KERNLE_AUTH_TOKEN": "",
            "KERNLE_USER_ID": "",
        }
        with patch.dict(os.environ, env):
            with patch("kernle.cli.commands.sync.get_kernle_home", return_value=creds_path):
                with patch.dict("sys.modules", {"httpx": mock_httpx}):
                    cmd_sync(_args(sync_action="status", json=True), k)

        output = json.loads(capsys.readouterr().out)
        assert output["backend_url"] == "https://legacy.api.com"
        assert output["authenticated"] is True

    def test_legacy_config_bad_json(self, k, capsys, tmp_path):
        """Handles corrupt config.json gracefully."""
        creds_path = tmp_path / "bad_config"
        creds_path.mkdir()
        (creds_path / "config.json").write_text("not valid json")

        mock_httpx = MagicMock()
        mock_httpx.get.return_value = _make_response(200)

        env = {
            "KERNLE_DATA_DIR": str(creds_path),
            "KERNLE_BACKEND_URL": "",
            "KERNLE_AUTH_TOKEN": "",
            "KERNLE_USER_ID": "",
        }
        with patch.dict(os.environ, env):
            with patch("kernle.cli.commands.sync.get_kernle_home", return_value=creds_path):
                with patch.dict("sys.modules", {"httpx": mock_httpx}):
                    cmd_sync(_args(sync_action="status", json=True), k)

        output = json.loads(capsys.readouterr().out)
        # Should proceed without crashing, backend_url should be unconfigured
        assert output["backend_url"] == "(not configured)"

    def test_credentials_bad_json(self, k, capsys, tmp_path):
        """Handles corrupt credentials.json gracefully."""
        creds_path = tmp_path / "bad_creds"
        creds_path.mkdir()
        (creds_path / "credentials.json").write_text("{bad json")

        mock_httpx = MagicMock()
        mock_httpx.get.return_value = _make_response(200)

        env = {
            "KERNLE_DATA_DIR": str(creds_path),
            "KERNLE_BACKEND_URL": "",
            "KERNLE_AUTH_TOKEN": "",
            "KERNLE_USER_ID": "",
        }
        with patch.dict(os.environ, env):
            with patch("kernle.cli.commands.sync.get_kernle_home", return_value=creds_path):
                with patch.dict("sys.modules", {"httpx": mock_httpx}):
                    cmd_sync(_args(sync_action="status", json=True), k)

        output = json.loads(capsys.readouterr().out)
        assert output["backend_url"] == "(not configured)"

    def test_credentials_token_field_alternatives(self, k, capsys, tmp_path):
        """Supports 'token' and 'api_key' field names as auth_token."""
        creds = {"backend_url": "https://api.test.com", "token": "tok-from-token-field"}
        creds_path = _setup_creds(tmp_path, creds=creds, dirname="alt_token")

        mock_httpx = MagicMock()
        mock_httpx.get.return_value = _make_response(200)

        with patch.dict(os.environ, {"KERNLE_DATA_DIR": str(creds_path)}):
            with patch("kernle.cli.commands.sync.get_kernle_home", return_value=creds_path):
                with patch.dict("sys.modules", {"httpx": mock_httpx}):
                    cmd_sync(_args(sync_action="status", json=True), k)

        output = json.loads(capsys.readouterr().out)
        assert output["authenticated"] is True


# ============================================================================
# httpx not installed (lines 88-90)
# ============================================================================


class TestHttpxNotInstalled:
    """Test behavior when httpx is not available."""

    def test_status_httpx_missing(self, k, tmp_path):
        """Exits when httpx is not installed."""
        creds_path = _setup_creds(tmp_path, dirname="no_httpx")

        env = {
            "KERNLE_DATA_DIR": str(creds_path),
            "KERNLE_BACKEND_URL": "",
            "KERNLE_AUTH_TOKEN": "",
            "KERNLE_USER_ID": "",
        }
        with patch.dict(os.environ, env):
            with patch("kernle.cli.commands.sync.get_kernle_home", return_value=creds_path):
                # Simulate httpx import failure
                import builtins

                original_import = builtins.__import__

                def mock_import(name, *args, **kwargs):
                    if name == "httpx":
                        raise ImportError("No module named 'httpx'")
                    return original_import(name, *args, **kwargs)

                with patch("builtins.__import__", side_effect=mock_import):
                    with pytest.raises(SystemExit) as exc_info:
                        cmd_sync(_args(sync_action="status"), k)
                    assert exc_info.value.code == 1


# ============================================================================
# Backend connection check (lines 92-108)
# ============================================================================


class TestBackendConnectionCheck:
    """Test backend health check paths."""

    def test_backend_connection_exception(self, k, capsys, tmp_path):
        """Connection exception returns False with error message."""
        creds = {"backend_url": "https://unreachable.api.com", "auth_token": "tok"}
        creds_path = _setup_creds(tmp_path, creds=creds, dirname="conn_err")

        mock_httpx = MagicMock()
        mock_httpx.get.side_effect = ConnectionError("Network unreachable")

        with patch.dict(os.environ, {"KERNLE_DATA_DIR": str(creds_path)}):
            with patch("kernle.cli.commands.sync.get_kernle_home", return_value=creds_path):
                with patch.dict("sys.modules", {"httpx": mock_httpx}):
                    cmd_sync(_args(sync_action="status", json=True), k)

        output = json.loads(capsys.readouterr().out)
        assert output["backend_connected"] is False
        assert "Connection failed" in output["connection_status"]

    def test_backend_no_auth_token(self, k, capsys, tmp_path):
        """Reports not authenticated when no token."""
        creds = {"backend_url": "https://api.test.com"}
        creds_path = _setup_creds(tmp_path, creds=creds, dirname="no_tok")

        mock_httpx = MagicMock()

        env = {"KERNLE_DATA_DIR": str(creds_path), "KERNLE_AUTH_TOKEN": ""}
        with patch.dict(os.environ, env):
            with patch("kernle.cli.commands.sync.get_kernle_home", return_value=creds_path):
                with patch.dict("sys.modules", {"httpx": mock_httpx}):
                    cmd_sync(_args(sync_action="status", json=True), k)

        output = json.loads(capsys.readouterr().out)
        assert output["backend_connected"] is False
        assert "Not authenticated" in output["connection_status"]


# ============================================================================
# Status display formatting (lines 160-208)
# ============================================================================


class TestStatusDisplayFormatting:
    """Test non-JSON status output with various conditions."""

    def _run_status(self, k, capsys, tmp_path, creds, dirname, **env_extra):
        """Helper to run status command with given creds."""
        creds_path = _setup_creds(tmp_path, creds=creds, dirname=dirname)

        mock_httpx = MagicMock()
        mock_httpx.get.return_value = _make_response(200)

        env = {"KERNLE_DATA_DIR": str(creds_path)}
        env.update(env_extra)
        with patch.dict(os.environ, env):
            with patch("kernle.cli.commands.sync.get_kernle_home", return_value=creds_path):
                with patch.dict("sys.modules", {"httpx": mock_httpx}):
                    cmd_sync(_args(sync_action="status"), k)
        return capsys.readouterr().out

    def test_status_with_user_id_connected(self, k, capsys, tmp_path):
        """Status display shows 'Synced as:' when connected with user_id."""
        creds = {"backend_url": "https://api.test.com", "auth_token": "tok", "user_id": "u42"}
        output = self._run_status(k, capsys, tmp_path, creds, "status_connected")
        assert "Synced as:" in output
        assert "u42/test-sync-cov" in output

    def test_status_with_user_id_disconnected(self, k, capsys, tmp_path):
        """Status display shows 'Will sync as:' when disconnected with user_id."""
        creds = {"backend_url": "https://api.test.com", "auth_token": "tok", "user_id": "u42"}
        creds_path = _setup_creds(tmp_path, creds=creds, dirname="status_disc")

        mock_httpx = MagicMock()
        mock_httpx.get.return_value = _make_response(503)

        with patch.dict(os.environ, {"KERNLE_DATA_DIR": str(creds_path)}):
            with patch("kernle.cli.commands.sync.get_kernle_home", return_value=creds_path):
                with patch.dict("sys.modules", {"httpx": mock_httpx}):
                    cmd_sync(_args(sync_action="status"), k)

        output = capsys.readouterr().out
        assert "Will sync as:" in output

    def test_status_shows_backend_url(self, k, capsys, tmp_path):
        """Status display includes backend URL."""
        creds = {"backend_url": "https://api.test.com", "auth_token": "tok"}
        output = self._run_status(k, capsys, tmp_path, creds, "status_url")
        assert "URL: https://api.test.com" in output

    def test_status_shows_user(self, k, capsys, tmp_path):
        """Status display includes user ID."""
        creds = {"backend_url": "https://api.test.com", "auth_token": "tok", "user_id": "u1"}
        output = self._run_status(k, capsys, tmp_path, creds, "status_user")
        assert "User: u1" in output

    def test_status_last_sync_just_now(self, k, capsys, tmp_path):
        """Status shows 'just now' for very recent sync."""
        # Set a recent last sync time
        k._storage._set_sync_meta("last_sync_time", k._storage._now())

        creds = {"backend_url": "https://api.test.com", "auth_token": "tok"}
        output = self._run_status(k, capsys, tmp_path, creds, "status_just_now")
        assert "just now" in output

    def test_status_last_sync_minutes_ago(self, k, capsys, tmp_path):
        """Status shows 'N minutes ago' for sync within the hour."""
        past = datetime.now(timezone.utc) - timedelta(minutes=15)
        k._storage._set_sync_meta("last_sync_time", past.isoformat())

        creds = {"backend_url": "https://api.test.com", "auth_token": "tok"}
        output = self._run_status(k, capsys, tmp_path, creds, "status_minutes")
        assert "minutes ago" in output

    def test_status_last_sync_hours_ago(self, k, capsys, tmp_path):
        """Status shows 'N hours ago' for sync within the day."""
        past = datetime.now(timezone.utc) - timedelta(hours=5)
        k._storage._set_sync_meta("last_sync_time", past.isoformat())

        creds = {"backend_url": "https://api.test.com", "auth_token": "tok"}
        output = self._run_status(k, capsys, tmp_path, creds, "status_hours")
        assert "hours ago" in output

    def test_status_last_sync_days_ago(self, k, capsys, tmp_path):
        """Status shows 'N days ago' for older sync."""
        past = datetime.now(timezone.utc) - timedelta(days=3)
        k._storage._set_sync_meta("last_sync_time", past.isoformat())

        creds = {"backend_url": "https://api.test.com", "auth_token": "tok"}
        output = self._run_status(k, capsys, tmp_path, creds, "status_days")
        assert "days ago" in output

    def test_status_suggestion_push(self, k, capsys, tmp_path):
        """Status shows push suggestion when pending ops and connected."""
        # Create a pending change
        from kernle.storage import Note

        k._storage.save_note(Note(id="n-sug", stack_id="test-sync-cov", content="pending"))

        creds = {"backend_url": "https://api.test.com", "auth_token": "tok"}
        output = self._run_status(k, capsys, tmp_path, creds, "status_push_sug")
        assert "kernle sync push" in output

    def test_status_suggestion_login(self, k, capsys, tmp_path):
        """Status shows login suggestion when not authenticated."""
        creds_path = _setup_creds(tmp_path, dirname="status_login_sug")

        mock_httpx = MagicMock()
        mock_httpx.get.return_value = _make_response(200)

        env = {
            "KERNLE_DATA_DIR": str(creds_path),
            "KERNLE_BACKEND_URL": "",
            "KERNLE_AUTH_TOKEN": "",
            "KERNLE_USER_ID": "",
        }
        with patch.dict(os.environ, env):
            with patch("kernle.cli.commands.sync.get_kernle_home", return_value=creds_path):
                with patch.dict("sys.modules", {"httpx": mock_httpx}):
                    cmd_sync(_args(sync_action="status"), k)

        output = capsys.readouterr().out
        assert "kernle auth login" in output

    def test_status_suggestion_check_connection(self, k, capsys, tmp_path):
        """Status shows connection check suggestion when backend down but authenticated."""
        creds = {"backend_url": "https://bad.api.com", "auth_token": "tok"}
        creds_path = _setup_creds(tmp_path, creds=creds, dirname="status_conn_sug")

        mock_httpx = MagicMock()
        mock_httpx.get.return_value = _make_response(503)

        with patch.dict(os.environ, {"KERNLE_DATA_DIR": str(creds_path)}):
            with patch("kernle.cli.commands.sync.get_kernle_home", return_value=creds_path):
                with patch.dict("sys.modules", {"httpx": mock_httpx}):
                    cmd_sync(_args(sync_action="status"), k)

        output = capsys.readouterr().out
        assert "Check backend" in output or "connection" in output.lower()


# ============================================================================
# Push orphan handling (lines 347-357, 362)
# ============================================================================


class TestPushOrphanHandling:
    """Test push path when source records are deleted (orphaned sync entries)."""

    def test_push_orphaned_entry_skipped(self, k, capsys, tmp_path):
        """Orphaned sync entries (no payload, no source record) are skipped."""
        # Manually insert a sync queue entry with no payload and a record
        # that doesn't exist in the source table
        with k._storage._connect() as conn:
            conn.execute(
                """INSERT INTO sync_queue (table_name, record_id, operation, local_updated_at, synced, payload)
                   VALUES (?, ?, ?, datetime('now'), 0, NULL)""",
                ("notes", "nonexistent-note", "upsert"),
            )
            conn.commit()

        creds = {"backend_url": "https://api.test.com", "auth_token": "tok"}
        creds_path = _setup_creds(tmp_path, creds=creds, dirname="push_orphan")

        push_resp = _make_response(200, json_data={"synced": 0, "conflicts": []})
        mock_httpx = MagicMock()
        mock_httpx.post.return_value = push_resp

        with patch.dict(os.environ, {"KERNLE_DATA_DIR": str(creds_path)}):
            with patch("kernle.cli.commands.sync.get_kernle_home", return_value=creds_path):
                with patch.dict("sys.modules", {"httpx": mock_httpx}):
                    cmd_sync(_args(sync_action="push"), k)

        output = capsys.readouterr().out
        assert "Skipped 1 orphaned entries" in output

    def test_push_bad_payload_no_source_record(self, k, capsys, tmp_path):
        """Push handles bad payload where source record is also missing (orphan path)."""
        # Insert a sync queue entry with bad payload AND no source record
        with k._storage._connect() as conn:
            conn.execute(
                """INSERT INTO sync_queue (table_name, record_id, operation, local_updated_at, synced, payload)
                   VALUES (?, ?, ?, datetime('now'), 0, ?)""",
                ("notes", "nonexistent-bad-pl", "upsert", "not valid json"),
            )
            conn.commit()

        creds = {"backend_url": "https://api.test.com", "auth_token": "tok"}
        creds_path = _setup_creds(tmp_path, creds=creds, dirname="push_bad_pl")

        push_resp = _make_response(200, json_data={"synced": 0, "conflicts": []})
        mock_httpx = MagicMock()
        mock_httpx.post.return_value = push_resp

        with patch.dict(os.environ, {"KERNLE_DATA_DIR": str(creds_path)}):
            with patch("kernle.cli.commands.sync.get_kernle_home", return_value=creds_path):
                with patch.dict("sys.modules", {"httpx": mock_httpx}):
                    cmd_sync(_args(sync_action="push"), k)

        output = capsys.readouterr().out
        # Bad payload + no source record = orphaned, should be skipped
        assert "Skipped 1 orphaned entries" in output

    def test_push_with_stored_payload(self, k, capsys, tmp_path):
        """Push uses stored payload when available, even if source is deleted."""
        payload = json.dumps({"id": "note-payload", "content": "from payload"})
        with k._storage._connect() as conn:
            conn.execute(
                """INSERT INTO sync_queue (table_name, record_id, operation, local_updated_at, synced, payload)
                   VALUES (?, ?, ?, datetime('now'), 0, ?)""",
                ("notes", "note-payload", "upsert", payload),
            )
            conn.commit()

        creds = {"backend_url": "https://api.test.com", "auth_token": "tok"}
        creds_path = _setup_creds(tmp_path, creds=creds, dirname="push_payload")

        push_resp = _make_response(200, json_data={"synced": 1, "conflicts": []})
        mock_httpx = MagicMock()
        mock_httpx.post.return_value = push_resp

        with patch.dict(os.environ, {"KERNLE_DATA_DIR": str(creds_path)}):
            with patch("kernle.cli.commands.sync.get_kernle_home", return_value=creds_path):
                with patch.dict("sys.modules", {"httpx": mock_httpx}):
                    cmd_sync(_args(sync_action="push"), k)

        output = capsys.readouterr().out
        assert "Pushed 1" in output
        # Verify the payload was sent
        call_args = mock_httpx.post.call_args
        sent_json = call_args[1]["json"]
        ops = sent_json["operations"]
        assert len(ops) == 1
        assert ops[0]["data"]["content"] == "from payload"


# ============================================================================
# Mixed operation types in a single push
# ============================================================================


class TestPushMixedOperationTypes:
    """Test push with multiple record types in one batch."""

    def test_push_mixed_record_types(self, k, capsys, tmp_path):
        """Push sends episodes, notes, and beliefs together in one payload."""
        from kernle.storage import Belief, Note

        # Create records of different types so they land in the sync queue
        k.episode(objective="Test mixed push", outcome="success", lessons=["learned"])
        k._storage.save_note(Note(id="n-mixed", stack_id="test-sync-cov", content="Mixed note"))
        k._storage.save_belief(
            Belief(
                id="b-mixed",
                stack_id="test-sync-cov",
                statement="Mixed belief",
                confidence=0.8,
            )
        )

        creds = {"backend_url": "https://api.test.com", "auth_token": "tok"}
        creds_path = _setup_creds(tmp_path, creds=creds, dirname="push_mixed")

        push_resp = _make_response(200, json_data={"synced": 3, "conflicts": []})
        mock_httpx = MagicMock()
        mock_httpx.post.return_value = push_resp

        with patch.dict(os.environ, {"KERNLE_DATA_DIR": str(creds_path)}):
            with patch("kernle.cli.commands.sync.get_kernle_home", return_value=creds_path):
                with patch.dict("sys.modules", {"httpx": mock_httpx}):
                    cmd_sync(_args(sync_action="push"), k)

        output = capsys.readouterr().out
        assert "Pushed 3 changes" in output

        # Verify the payload contains all 3 record types
        call_args = mock_httpx.post.call_args
        sent_json = call_args[1]["json"]
        ops = sent_json["operations"]
        tables_pushed = {op["table"] for op in ops}
        assert "episodes" in tables_pushed
        assert "notes" in tables_pushed
        assert "beliefs" in tables_pushed

        # Verify each operation has data
        for op in ops:
            assert "data" in op, f"Operation for {op['table']} missing data"
            assert op["operation"] == "update"


# ============================================================================
# Pull delete operation (line 479)
# ============================================================================


class TestPullDeleteOperation:
    """Test pull with delete operations."""

    def test_pull_delete_operation(self, k, capsys, tmp_path):
        """Pull handles delete operations (delete is a no-op pass in current impl)."""
        creds = {"backend_url": "https://api.test.com", "auth_token": "tok"}
        creds_path = _setup_creds(tmp_path, creds=creds, dirname="pull_del")

        ops = [{"table": "notes", "record_id": "deleted-note", "operation": "delete"}]
        pull_resp = _make_response(200, json_data={"operations": ops, "has_more": False})
        mock_httpx = MagicMock()
        mock_httpx.post.return_value = pull_resp

        with patch.dict(os.environ, {"KERNLE_DATA_DIR": str(creds_path)}):
            with patch("kernle.cli.commands.sync.get_kernle_home", return_value=creds_path):
                with patch.dict("sys.modules", {"httpx": mock_httpx}):
                    cmd_sync(_args(sync_action="pull"), k)

        output = capsys.readouterr().out
        # Delete is handled as a pass (no-op), so applied count is 0
        assert "Pulled 0 changes" in output


# ============================================================================
# Pull conflict display (lines 531-533, 556)
# ============================================================================


class TestPullConflictDisplay:
    """Test pull output when there are conflicts during apply."""

    def test_pull_with_apply_conflicts(self, k, capsys, tmp_path):
        """Pull reports conflicts when record apply fails."""
        creds = {"backend_url": "https://api.test.com", "auth_token": "tok", "user_id": "u1"}
        creds_path = _setup_creds(tmp_path, creds=creds, dirname="pull_conf_disp")

        # Create an episode operation with data that will fail during apply
        ops = [
            {
                "table": "episodes",
                "record_id": "ep-bad",
                "operation": "upsert",
                "data": None,  # Missing required data - will cause exception
            }
        ]
        pull_resp = _make_response(200, json_data={"operations": ops, "has_more": False})
        mock_httpx = MagicMock()
        mock_httpx.post.return_value = pull_resp

        with patch.dict(os.environ, {"KERNLE_DATA_DIR": str(creds_path)}):
            with patch("kernle.cli.commands.sync.get_kernle_home", return_value=creds_path):
                with patch.dict("sys.modules", {"httpx": mock_httpx}):
                    cmd_sync(_args(sync_action="pull"), k)

        output = capsys.readouterr().out
        assert "Pulled" in output

    def test_pull_with_user_id_display(self, k, capsys, tmp_path):
        """Pull shows 'From: user/project' when user_id is set."""
        creds = {"backend_url": "https://api.test.com", "auth_token": "tok", "user_id": "u99"}
        creds_path = _setup_creds(tmp_path, creds=creds, dirname="pull_uid")

        ops = [{"table": "other", "record_id": "x1", "operation": "upsert", "data": {"a": "b"}}]
        pull_resp = _make_response(200, json_data={"operations": ops, "has_more": False})
        mock_httpx = MagicMock()
        mock_httpx.post.return_value = pull_resp

        with patch.dict(os.environ, {"KERNLE_DATA_DIR": str(creds_path)}):
            with patch("kernle.cli.commands.sync.get_kernle_home", return_value=creds_path):
                with patch.dict("sys.modules", {"httpx": mock_httpx}):
                    cmd_sync(_args(sync_action="pull"), k)

        output = capsys.readouterr().out
        assert "From: u99/test-sync-cov" in output

    def test_pull_full_shows_message(self, k, capsys, tmp_path):
        """Pull with --full shows '(full)' in output."""
        creds = {"backend_url": "https://api.test.com", "auth_token": "tok"}
        creds_path = _setup_creds(tmp_path, creds=creds, dirname="pull_full_msg")

        pull_resp = _make_response(200, json_data={"operations": [], "has_more": False})
        mock_httpx = MagicMock()
        mock_httpx.post.return_value = pull_resp

        with patch.dict(os.environ, {"KERNLE_DATA_DIR": str(creds_path)}):
            with patch("kernle.cli.commands.sync.get_kernle_home", return_value=creds_path):
                with patch.dict("sys.modules", {"httpx": mock_httpx}):
                    cmd_sync(_args(sync_action="pull", full=True), k)

        output = capsys.readouterr().out
        assert "(full)" in output


# ============================================================================
# Full sync push paths (lines 617-784)
# ============================================================================


class TestFullSyncPushPaths:
    """Test the full sync push code paths."""

    def _setup_full_sync(self, k, tmp_path, dirname, pull_resp, push_resp=None):
        """Helper for full sync tests."""
        creds = {"backend_url": "https://api.test.com", "auth_token": "tok", "user_id": "u1"}
        creds_path = _setup_creds(tmp_path, creds=creds, dirname=dirname)

        call_count = {"n": 0}

        def route_post(*args, **kwargs):
            call_count["n"] += 1
            if call_count["n"] == 1:
                return pull_resp
            return push_resp or _make_response(200, json_data={"synced": 0, "conflicts": []})

        mock_httpx = MagicMock()
        mock_httpx.post.side_effect = route_post

        return creds_path, mock_httpx

    def test_full_sync_with_user_id_display(self, k, capsys, tmp_path):
        """Full sync shows 'Syncing as:' when user_id is set."""
        pull_resp = _make_response(200, json_data={"operations": [], "has_more": False})
        creds_path, mock_httpx = self._setup_full_sync(k, tmp_path, "full_uid", pull_resp)

        with patch.dict(os.environ, {"KERNLE_DATA_DIR": str(creds_path)}):
            with patch("kernle.cli.commands.sync.get_kernle_home", return_value=creds_path):
                with patch.dict("sys.modules", {"httpx": mock_httpx}):
                    cmd_sync(_args(sync_action="full"), k)

        output = capsys.readouterr().out
        assert "Syncing as:" in output

    def test_full_sync_pull_failure(self, k, capsys, tmp_path):
        """Full sync continues even if pull returns non-200."""
        creds = {"backend_url": "https://api.test.com", "auth_token": "tok", "user_id": "u1"}
        creds_path = _setup_creds(tmp_path, creds=creds, dirname="full_pull_fail")

        mock_httpx = MagicMock()
        mock_httpx.post.return_value = _make_response(500, text="Server Error")

        with patch.dict(os.environ, {"KERNLE_DATA_DIR": str(creds_path)}):
            with patch("kernle.cli.commands.sync.get_kernle_home", return_value=creds_path):
                with patch.dict("sys.modules", {"httpx": mock_httpx}):
                    cmd_sync(_args(sync_action="full"), k)

        output = capsys.readouterr().out
        assert "Pull returned status 500" in output
        assert "Full sync complete" in output

    def test_full_sync_pull_exception(self, k, capsys, tmp_path):
        """Full sync continues even if pull raises exception."""
        creds = {"backend_url": "https://api.test.com", "auth_token": "tok", "user_id": "u1"}
        creds_path = _setup_creds(tmp_path, creds=creds, dirname="full_pull_exc")

        call_count = {"n": 0}

        def route_post(*args, **kwargs):
            call_count["n"] += 1
            if call_count["n"] == 1:
                raise ConnectionError("Network down")
            return _make_response(200, json_data={"synced": 0, "conflicts": []})

        mock_httpx = MagicMock()
        mock_httpx.post.side_effect = route_post

        with patch.dict(os.environ, {"KERNLE_DATA_DIR": str(creds_path)}):
            with patch("kernle.cli.commands.sync.get_kernle_home", return_value=creds_path):
                with patch.dict("sys.modules", {"httpx": mock_httpx}):
                    cmd_sync(_args(sync_action="full"), k)

        output = capsys.readouterr().out
        assert "Pull failed" in output
        assert "Full sync complete" in output

    def test_full_sync_push_failure(self, k, capsys, tmp_path):
        """Full sync reports push failure status code."""
        from kernle.storage import Note

        k._storage.save_note(Note(id="n-full", stack_id="test-sync-cov", content="data"))

        pull_resp = _make_response(200, json_data={"operations": [], "has_more": False})
        push_resp = _make_response(500, text="Server Error")
        creds_path, mock_httpx = self._setup_full_sync(
            k, tmp_path, "full_push_fail", pull_resp, push_resp
        )

        with patch.dict(os.environ, {"KERNLE_DATA_DIR": str(creds_path)}):
            with patch("kernle.cli.commands.sync.get_kernle_home", return_value=creds_path):
                with patch.dict("sys.modules", {"httpx": mock_httpx}):
                    cmd_sync(_args(sync_action="full"), k)

        output = capsys.readouterr().out
        assert "Push returned status 500" in output

    def test_full_sync_push_exception(self, k, capsys, tmp_path):
        """Full sync handles push network error."""
        from kernle.storage import Note

        k._storage.save_note(Note(id="n-full-exc", stack_id="test-sync-cov", content="data"))

        creds = {"backend_url": "https://api.test.com", "auth_token": "tok"}
        creds_path = _setup_creds(tmp_path, creds=creds, dirname="full_push_exc")

        call_count = {"n": 0}

        def route_post(*args, **kwargs):
            call_count["n"] += 1
            if call_count["n"] == 1:
                return _make_response(200, json_data={"operations": [], "has_more": False})
            raise ConnectionError("Push network error")

        mock_httpx = MagicMock()
        mock_httpx.post.side_effect = route_post

        with patch.dict(os.environ, {"KERNLE_DATA_DIR": str(creds_path)}):
            with patch("kernle.cli.commands.sync.get_kernle_home", return_value=creds_path):
                with patch.dict("sys.modules", {"httpx": mock_httpx}):
                    cmd_sync(_args(sync_action="full"), k)

        output = capsys.readouterr().out
        assert "Push failed" in output

    def test_full_sync_remaining_ops(self, k, capsys, tmp_path):
        """Full sync shows remaining operations count."""
        from kernle.storage import Note

        k._storage.save_note(Note(id="n-remain1", stack_id="test-sync-cov", content="a"))
        k._storage.save_note(Note(id="n-remain2", stack_id="test-sync-cov", content="b"))

        creds = {"backend_url": "https://api.test.com", "auth_token": "tok"}
        creds_path = _setup_creds(tmp_path, creds=creds, dirname="full_remain")

        call_count = {"n": 0}

        def route_post(*args, **kwargs):
            call_count["n"] += 1
            if call_count["n"] == 1:
                return _make_response(200, json_data={"operations": [], "has_more": False})
            # Report synced=0 so all remain pending
            return _make_response(200, json_data={"synced": 0, "conflicts": []})

        mock_httpx = MagicMock()
        mock_httpx.post.side_effect = route_post

        with patch.dict(os.environ, {"KERNLE_DATA_DIR": str(creds_path)}):
            with patch("kernle.cli.commands.sync.get_kernle_home", return_value=creds_path):
                with patch.dict("sys.modules", {"httpx": mock_httpx}):
                    cmd_sync(_args(sync_action="full"), k)

        output = capsys.readouterr().out
        assert "2 operations still pending" in output

    def test_full_sync_orphaned_entries(self, k, capsys, tmp_path):
        """Full sync skips orphaned entries in push phase."""
        # Create an orphaned sync queue entry
        with k._storage._connect() as conn:
            conn.execute(
                """INSERT INTO sync_queue (table_name, record_id, operation, local_updated_at, synced, payload)
                   VALUES (?, ?, ?, datetime('now'), 0, NULL)""",
                ("notes", "nonexistent", "upsert"),
            )
            conn.commit()

        pull_resp = _make_response(200, json_data={"operations": [], "has_more": False})
        creds_path, mock_httpx = self._setup_full_sync(k, tmp_path, "full_orphan", pull_resp)

        with patch.dict(os.environ, {"KERNLE_DATA_DIR": str(creds_path)}):
            with patch("kernle.cli.commands.sync.get_kernle_home", return_value=creds_path):
                with patch.dict("sys.modules", {"httpx": mock_httpx}):
                    cmd_sync(_args(sync_action="full"), k)

        output = capsys.readouterr().out
        assert "Skipped 1 orphaned entries" in output

    def test_full_sync_push_with_payload(self, k, capsys, tmp_path):
        """Full sync push uses stored payload when available."""
        payload = json.dumps({"id": "note-fp", "content": "from payload"})
        with k._storage._connect() as conn:
            conn.execute(
                """INSERT INTO sync_queue (table_name, record_id, operation, local_updated_at, synced, payload)
                   VALUES (?, ?, ?, datetime('now'), 0, ?)""",
                ("notes", "note-fp", "upsert", payload),
            )
            conn.commit()

        pull_resp = _make_response(200, json_data={"operations": [], "has_more": False})
        push_resp = _make_response(200, json_data={"synced": 1, "conflicts": []})
        creds_path, mock_httpx = self._setup_full_sync(
            k, tmp_path, "full_payload", pull_resp, push_resp
        )

        with patch.dict(os.environ, {"KERNLE_DATA_DIR": str(creds_path)}):
            with patch("kernle.cli.commands.sync.get_kernle_home", return_value=creds_path):
                with patch.dict("sys.modules", {"httpx": mock_httpx}):
                    cmd_sync(_args(sync_action="full"), k)

        output = capsys.readouterr().out
        assert "Pushed 1 changes" in output


# ============================================================================
# format_datetime string pass-through (line 122)
# ============================================================================


class TestFormatDatetimeStringPassthrough:
    """Test format_datetime when dt is already a string."""

    def test_pull_with_string_last_sync_time(self, k, capsys, tmp_path):
        """When last_sync_time is stored as a string, format_datetime returns it unchanged."""
        # Set last sync time as an ISO string — get_last_sync_time will parse it,
        # but we can mock it to return a string to hit the isinstance(dt, str) branch.
        creds = {"backend_url": "https://api.test.com", "auth_token": "tok"}
        creds_path = _setup_creds(tmp_path, creds=creds, dirname="fmt_str")

        pull_resp = _make_response(200, json_data={"operations": [], "has_more": False})
        mock_httpx = MagicMock()
        mock_httpx.post.return_value = pull_resp

        # Mock get_last_sync_time to return a string (triggers line 122)
        original_get_last = k._storage.get_last_sync_time

        def mock_get_last():
            return "2024-01-15T10:30:00"

        k._storage.get_last_sync_time = mock_get_last

        with patch.dict(os.environ, {"KERNLE_DATA_DIR": str(creds_path)}):
            with patch("kernle.cli.commands.sync.get_kernle_home", return_value=creds_path):
                with patch.dict("sys.modules", {"httpx": mock_httpx}):
                    cmd_sync(_args(sync_action="pull"), k)

        k._storage.get_last_sync_time = original_get_last

        # Verify the string was passed through to the request
        call_args = mock_httpx.post.call_args
        sent_json = call_args[1]["json"]
        assert sent_json["since"] == "2024-01-15T10:30:00"


# ============================================================================
# Naive datetime in status display (lines 184-186)
# ============================================================================


class TestNaiveDatetimeStatus:
    """Test status display when last_sync is a naive datetime (no tzinfo)."""

    def test_status_naive_datetime_gets_utc(self, k, capsys, tmp_path):
        """Naive datetime without tzinfo gets UTC attached for elapsed calculation."""
        creds = {"backend_url": "https://api.test.com", "auth_token": "tok"}
        creds_path = _setup_creds(tmp_path, creds=creds, dirname="naive_dt")

        mock_httpx = MagicMock()
        mock_httpx.get.return_value = _make_response(200)

        # Return a naive datetime (no tzinfo) to trigger lines 184-186
        naive_dt = datetime(2024, 1, 15, 10, 30, 0)  # no timezone
        original_get_last = k._storage.get_last_sync_time
        k._storage.get_last_sync_time = lambda: naive_dt

        with patch.dict(os.environ, {"KERNLE_DATA_DIR": str(creds_path)}):
            with patch("kernle.cli.commands.sync.get_kernle_home", return_value=creds_path):
                with patch.dict("sys.modules", {"httpx": mock_httpx}):
                    cmd_sync(_args(sync_action="status"), k)

        k._storage.get_last_sync_time = original_get_last

        output = capsys.readouterr().out
        # Should show days ago since 2024-01-15 is far in the past
        assert "days ago" in output


# ============================================================================
# Push with live record fallback — dataclass extraction (lines 288-297)
# ============================================================================


class TestPushLiveRecordFallback:
    """Test push when payload is missing but source record exists in DB."""

    def test_push_uses_live_record_when_no_payload(self, k, capsys, tmp_path):
        """Push extracts dataclass fields from live record when no stored payload."""
        from kernle.storage import Note

        # Save a note so it exists in the source table
        k._storage.save_note(Note(id="n-live", stack_id="test-sync-cov", content="live content"))

        # Clear both payload and data columns so it must use the live record
        with k._storage._connect() as conn:
            conn.execute(
                "UPDATE sync_queue SET payload = NULL, data = NULL WHERE record_id = ?",
                ("n-live",),
            )
            conn.commit()

        creds = {"backend_url": "https://api.test.com", "auth_token": "tok"}
        creds_path = _setup_creds(tmp_path, creds=creds, dirname="push_live")

        push_resp = _make_response(200, json_data={"synced": 1, "conflicts": []})
        mock_httpx = MagicMock()
        mock_httpx.post.return_value = push_resp

        with patch.dict(os.environ, {"KERNLE_DATA_DIR": str(creds_path)}):
            with patch("kernle.cli.commands.sync.get_kernle_home", return_value=creds_path):
                with patch.dict("sys.modules", {"httpx": mock_httpx}):
                    cmd_sync(_args(sync_action="push"), k)

        output = capsys.readouterr().out
        assert "Pushed 1 changes" in output

        # Verify the data was extracted from the live record
        call_args = mock_httpx.post.call_args
        sent_json = call_args[1]["json"]
        ops = sent_json["operations"]
        assert len(ops) == 1
        assert ops[0]["data"]["content"] == "live content"
        assert ops[0]["data"]["id"] == "n-live"


# ============================================================================
# Push response with conflicts (lines 354-359)
# ============================================================================


class TestPushResponseConflicts:
    """Test push output when backend reports conflicts."""

    def test_push_with_conflicts_display(self, k, capsys, tmp_path):
        """Push shows conflict details when backend reports them."""
        from kernle.storage import Note

        k._storage.save_note(Note(id="n-conf", stack_id="test-sync-cov", content="conflict data"))

        creds = {"backend_url": "https://api.test.com", "auth_token": "tok", "user_id": "u1"}
        creds_path = _setup_creds(tmp_path, creds=creds, dirname="push_conf")

        conflicts_list = [
            {"record_id": "n-conf", "error": "version mismatch"},
            {"record_id": "n-other", "error": "stale data"},
        ]
        push_resp = _make_response(200, json_data={"synced": 1, "conflicts": conflicts_list})
        mock_httpx = MagicMock()
        mock_httpx.post.return_value = push_resp

        with patch.dict(os.environ, {"KERNLE_DATA_DIR": str(creds_path)}):
            with patch("kernle.cli.commands.sync.get_kernle_home", return_value=creds_path):
                with patch.dict("sys.modules", {"httpx": mock_httpx}):
                    cmd_sync(_args(sync_action="push"), k)

        output = capsys.readouterr().out
        assert "2 conflicts:" in output
        assert "n-conf" in output
        assert "version mismatch" in output
        assert "Synced as: u1/test-sync-cov" in output


# ============================================================================
# Pull with since parameter (line 399) and conflict + has_more display
# ============================================================================


class TestPullWithSinceAndConflicts:
    """Test pull with incremental since time and conflict/has_more branches."""

    def test_pull_incremental_with_since(self, k, capsys, tmp_path):
        """Pull sends since parameter when last_sync_time exists and full=False."""
        # Set a last sync time so since is non-None
        k._storage._set_sync_meta("last_sync_time", "2024-06-01T12:00:00+00:00")

        creds = {"backend_url": "https://api.test.com", "auth_token": "tok"}
        creds_path = _setup_creds(tmp_path, creds=creds, dirname="pull_since")

        pull_resp = _make_response(200, json_data={"operations": [], "has_more": False})
        mock_httpx = MagicMock()
        mock_httpx.post.return_value = pull_resp

        with patch.dict(os.environ, {"KERNLE_DATA_DIR": str(creds_path)}):
            with patch("kernle.cli.commands.sync.get_kernle_home", return_value=creds_path):
                with patch.dict("sys.modules", {"httpx": mock_httpx}):
                    cmd_sync(_args(sync_action="pull", full=False), k)

        # Verify since was included in the request
        call_args = mock_httpx.post.call_args
        sent_json = call_args[1]["json"]
        assert "since" in sent_json

    def test_pull_exception_during_apply_counts_conflict(self, k, capsys, tmp_path):
        """Pull counts exceptions during record apply as conflicts."""
        creds = {"backend_url": "https://api.test.com", "auth_token": "tok", "user_id": "u1"}
        creds_path = _setup_creds(tmp_path, creds=creds, dirname="pull_exc")

        # Create episode operation that will raise during save_episode
        ops = [
            {
                "table": "episodes",
                "record_id": "ep-crash",
                "operation": "upsert",
                "data": {"objective": "test", "outcome_type": "success"},
            }
        ]
        pull_resp = _make_response(200, json_data={"operations": ops, "has_more": True})
        mock_httpx = MagicMock()
        mock_httpx.post.return_value = pull_resp

        # Make save_episode raise to trigger the except path
        original_save = k._storage.save_episode
        k._storage.save_episode = MagicMock(side_effect=ValueError("bad episode"))

        with patch.dict(os.environ, {"KERNLE_DATA_DIR": str(creds_path)}):
            with patch("kernle.cli.commands.sync.get_kernle_home", return_value=creds_path):
                with patch.dict("sys.modules", {"httpx": mock_httpx}):
                    cmd_sync(_args(sync_action="pull"), k)

        k._storage.save_episode = original_save

        output = capsys.readouterr().out
        assert "1 conflicts during apply" in output
        assert "More changes available" in output
        assert "From: u1/test-sync-cov" in output


# ============================================================================
# Full sync push with bad payload + live record fallback (lines 615-616, 623-633)
# ============================================================================


class TestFullSyncBadPayloadLiveRecord:
    """Test full sync push when payload is invalid JSON but live record exists."""

    def test_full_sync_bad_payload_falls_back_to_live_record(self, k, capsys, tmp_path):
        """Full sync push falls back to live record when payload is bad JSON."""
        from kernle.storage import Note

        # Save a note so it exists in the source table
        k._storage.save_note(Note(id="n-badpl", stack_id="test-sync-cov", content="live fallback"))

        # Corrupt both payload and data in the sync queue so bad JSON hits the except
        # path and then falls through to the live record lookup
        with k._storage._connect() as conn:
            conn.execute(
                "UPDATE sync_queue SET payload = ?, data = ? WHERE record_id = ?",
                ("not valid json", "not valid json", "n-badpl"),
            )
            conn.commit()

        creds = {"backend_url": "https://api.test.com", "auth_token": "tok"}
        creds_path = _setup_creds(tmp_path, creds=creds, dirname="full_badpl")

        call_count = {"n": 0}

        def route_post(*args, **kwargs):
            call_count["n"] += 1
            if call_count["n"] == 1:
                return _make_response(200, json_data={"operations": [], "has_more": False})
            return _make_response(200, json_data={"synced": 1, "conflicts": []})

        mock_httpx = MagicMock()
        mock_httpx.post.side_effect = route_post

        with patch.dict(os.environ, {"KERNLE_DATA_DIR": str(creds_path)}):
            with patch("kernle.cli.commands.sync.get_kernle_home", return_value=creds_path):
                with patch.dict("sys.modules", {"httpx": mock_httpx}):
                    cmd_sync(_args(sync_action="full"), k)

        output = capsys.readouterr().out
        assert "Pushed 1 changes" in output

        # Verify the live record data was sent (from dataclass extraction)
        push_call = mock_httpx.post.call_args_list[1]
        sent_json = push_call[1]["json"]
        ops = sent_json["operations"]
        assert len(ops) == 1
        assert ops[0]["data"]["content"] == "live fallback"


# ============================================================================
# Conflicts display with local/cloud summaries (lines 733-737)
# ============================================================================


class TestConflictsDisplaySummaries:
    """Test conflict history display with local and cloud summaries."""

    def test_conflicts_display_with_summaries(self, k, capsys, tmp_path):
        """Conflict display shows local and cloud summaries when present."""
        from kernle.types import SyncConflict

        # Save a conflict record with both summaries
        conflict = SyncConflict(
            id="conf-1",
            table="notes",
            record_id="n-conflict-123456",
            local_version={"content": "local version"},
            cloud_version={"content": "cloud version"},
            resolution="cloud_wins",
            resolved_at=datetime(2024, 6, 15, 14, 30, 0, tzinfo=timezone.utc),
            local_summary="My local note",
            cloud_summary="Remote updated note",
        )
        k._storage.save_sync_conflict(conflict)

        creds = {"backend_url": "https://api.test.com", "auth_token": "tok"}
        creds_path = _setup_creds(tmp_path, creds=creds, dirname="conf_summ")

        mock_httpx = MagicMock()
        mock_httpx.get.return_value = _make_response(200)

        with patch.dict(os.environ, {"KERNLE_DATA_DIR": str(creds_path)}):
            with patch("kernle.cli.commands.sync.get_kernle_home", return_value=creds_path):
                with patch.dict("sys.modules", {"httpx": mock_httpx}):
                    cmd_sync(_args(sync_action="conflicts"), k)

        output = capsys.readouterr().out
        assert "Sync Conflict History" in output
        assert "cloud wins" in output
        assert 'Local:  "My local note"' in output
        assert 'Cloud:  "Remote updated note"' in output
        assert "n-confli..." in output
        assert "2024-06-15 14:30" in output

    def test_conflicts_display_without_summaries(self, k, capsys, tmp_path):
        """Conflict display handles missing summaries (neither local nor cloud)."""
        from kernle.types import SyncConflict

        conflict = SyncConflict(
            id="conf-2",
            table="episodes",
            record_id="ep-nosumm-12345678",
            local_version={"objective": "local"},
            cloud_version={"objective": "cloud"},
            resolution="local_wins",
            resolved_at=datetime(2024, 7, 20, 8, 0, 0, tzinfo=timezone.utc),
            local_summary=None,
            cloud_summary=None,
        )
        k._storage.save_sync_conflict(conflict)

        creds = {"backend_url": "https://api.test.com", "auth_token": "tok"}
        creds_path = _setup_creds(tmp_path, creds=creds, dirname="conf_nosumm")

        mock_httpx = MagicMock()
        mock_httpx.get.return_value = _make_response(200)

        with patch.dict(os.environ, {"KERNLE_DATA_DIR": str(creds_path)}):
            with patch("kernle.cli.commands.sync.get_kernle_home", return_value=creds_path):
                with patch.dict("sys.modules", {"httpx": mock_httpx}):
                    cmd_sync(_args(sync_action="conflicts"), k)

        output = capsys.readouterr().out
        assert "local wins" in output
        assert "Local:" not in output  # No summary means no "Local:" line
        assert "Cloud:" not in output


# ============================================================================
# Push/full-sync with delete operations (branch 270->311, 609->646)
# ============================================================================


class TestPushDeleteOperations:
    """Test push and full sync with delete operations in the sync queue."""

    def test_push_delete_operation_skips_record_extraction(self, k, capsys, tmp_path):
        """Push with delete operation skips record data extraction entirely."""
        # Insert a delete operation into the sync queue
        with k._storage._connect() as conn:
            conn.execute(
                """INSERT INTO sync_queue (table_name, record_id, operation, local_updated_at, synced)
                   VALUES (?, ?, ?, datetime('now'), 0)""",
                ("notes", "n-deleted", "delete"),
            )
            conn.commit()

        creds = {"backend_url": "https://api.test.com", "auth_token": "tok"}
        creds_path = _setup_creds(tmp_path, creds=creds, dirname="push_del")

        push_resp = _make_response(200, json_data={"synced": 1, "conflicts": []})
        mock_httpx = MagicMock()
        mock_httpx.post.return_value = push_resp

        with patch.dict(os.environ, {"KERNLE_DATA_DIR": str(creds_path)}):
            with patch("kernle.cli.commands.sync.get_kernle_home", return_value=creds_path):
                with patch.dict("sys.modules", {"httpx": mock_httpx}):
                    cmd_sync(_args(sync_action="push"), k)

        output = capsys.readouterr().out
        assert "Pushed 1 changes" in output

        # Verify the delete operation was sent without data field
        call_args = mock_httpx.post.call_args
        sent_json = call_args[1]["json"]
        ops = sent_json["operations"]
        assert len(ops) == 1
        assert ops[0]["operation"] == "delete"
        assert "data" not in ops[0]

    def test_full_sync_delete_operation(self, k, capsys, tmp_path):
        """Full sync push with delete operation skips record extraction."""
        # Insert a delete operation
        with k._storage._connect() as conn:
            conn.execute(
                """INSERT INTO sync_queue (table_name, record_id, operation, local_updated_at, synced)
                   VALUES (?, ?, ?, datetime('now'), 0)""",
                ("notes", "n-full-del", "delete"),
            )
            conn.commit()

        creds = {"backend_url": "https://api.test.com", "auth_token": "tok"}
        creds_path = _setup_creds(tmp_path, creds=creds, dirname="full_del")

        call_count = {"n": 0}

        def route_post(*args, **kwargs):
            call_count["n"] += 1
            if call_count["n"] == 1:
                return _make_response(200, json_data={"operations": [], "has_more": False})
            return _make_response(200, json_data={"synced": 1, "conflicts": []})

        mock_httpx = MagicMock()
        mock_httpx.post.side_effect = route_post

        with patch.dict(os.environ, {"KERNLE_DATA_DIR": str(creds_path)}):
            with patch("kernle.cli.commands.sync.get_kernle_home", return_value=creds_path):
                with patch.dict("sys.modules", {"httpx": mock_httpx}):
                    cmd_sync(_args(sync_action="full"), k)

        output = capsys.readouterr().out
        assert "Pushed 1 changes" in output

        # Verify delete sent without data
        push_call = mock_httpx.post.call_args_list[1]
        sent_json = push_call[1]["json"]
        ops = sent_json["operations"]
        assert len(ops) == 1
        assert ops[0]["operation"] == "delete"
        assert "data" not in ops[0]
