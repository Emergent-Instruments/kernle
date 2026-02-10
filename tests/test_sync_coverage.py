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
        assert "Check backend" in output or "kernle auth login" in output


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
        assert "orphaned" in output.lower() or "Skipped" in output

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
        assert "orphaned" in output.lower() or "Skipped" in output

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
        assert "conflicts" in output.lower() or "Pulled" in output

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
        assert "still pending" in output

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
        assert "orphaned" in output.lower() or "Skipped" in output

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
