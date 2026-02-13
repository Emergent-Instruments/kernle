"""Tests for kernle.cli.commands.auth — cmd_auth and cmd_auth_keys functions."""

import argparse
import json
import os
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, patch

import pytest

from kernle import Kernle
from kernle.cli.commands.auth import cmd_auth, cmd_auth_keys

# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def storage(tmp_path, sqlite_storage_factory):
    return sqlite_storage_factory(stack_id="test-auth", db_path=tmp_path / "auth.db")


@pytest.fixture
def k(storage):
    return Kernle(stack_id="test-auth", storage=storage, strict=False)


def _args(**kwargs):
    """Build an argparse.Namespace with defaults for auth commands."""
    defaults = {
        "command": "auth",
        "auth_action": "status",
        "json": False,
        "backend_url": None,
        "email": None,
        "api_key": None,
        "force": False,
    }
    defaults.update(kwargs)
    return argparse.Namespace(**defaults)


def _make_response(status_code=200, json_data=None, text=""):
    """Create a mock HTTP response."""
    resp = MagicMock()
    resp.status_code = status_code
    resp.json.return_value = json_data or {}
    resp.text = text
    return resp


def _extract_json(output: str) -> dict:
    """Extract the first JSON object from mixed stdout output."""
    start = output.index("{")
    return json.loads(output[start:])


# ============================================================================
# auth status
# ============================================================================


class TestAuthStatus:
    """Tests for `kernle auth status`."""

    def test_status_no_credentials(self, k, capsys, tmp_path):
        """Shows not authenticated when no credentials file."""
        with patch.dict(os.environ, {"KERNLE_DATA_DIR": str(tmp_path)}):
            cmd_auth(_args(auth_action="status"), k)

        captured = capsys.readouterr().out
        assert "Not authenticated" in captured

    def test_status_no_credentials_json(self, k, capsys, tmp_path):
        """JSON output for no credentials."""
        with patch.dict(os.environ, {"KERNLE_DATA_DIR": str(tmp_path)}):
            cmd_auth(_args(auth_action="status", json=True), k)

        output = json.loads(capsys.readouterr().out)
        assert output["authenticated"] is False

    def test_status_with_valid_token(self, k, capsys, tmp_path):
        """Shows authenticated with valid (non-expired) token."""
        future = (datetime.now(timezone.utc) + timedelta(days=7)).isoformat()
        creds = {
            "user_id": "u123",
            "api_key": "test-key",
            "backend_url": "https://api.test.com",
            "auth_token": "jwt-token-here",
            "token_expires": future,
        }
        (tmp_path / "credentials.json").write_text(json.dumps(creds))

        with patch.dict(os.environ, {"KERNLE_DATA_DIR": str(tmp_path)}):
            cmd_auth(_args(auth_action="status"), k)

        captured = capsys.readouterr().out
        assert "u123" in captured
        assert "https://api.test.com" in captured
        assert "Valid" in captured

    def test_status_with_expired_token(self, k, capsys, tmp_path):
        """Shows token expired when token_expires is in the past."""
        past = (datetime.now(timezone.utc) - timedelta(hours=1)).isoformat()
        creds = {
            "user_id": "u456",
            "api_key": "test-expired-key",
            "backend_url": "https://api.test.com",
            "auth_token": "old-token",
            "token_expires": past,
        }
        (tmp_path / "credentials.json").write_text(json.dumps(creds))

        with patch.dict(os.environ, {"KERNLE_DATA_DIR": str(tmp_path)}):
            cmd_auth(_args(auth_action="status"), k)

        captured = capsys.readouterr().out
        assert "Expired" in captured

    def test_status_json_with_token(self, k, capsys, tmp_path):
        """JSON output includes all auth fields."""
        future = (datetime.now(timezone.utc) + timedelta(hours=2)).isoformat()
        creds = {
            "user_id": "u789",
            "api_key": "fake-jsontest",
            "backend_url": "https://api.test.com",
            "auth_token": "jwt-tok",
            "token_expires": future,
        }
        (tmp_path / "credentials.json").write_text(json.dumps(creds))

        with patch.dict(os.environ, {"KERNLE_DATA_DIR": str(tmp_path)}):
            cmd_auth(_args(auth_action="status", json=True), k)

        output = json.loads(capsys.readouterr().out)
        assert output["authenticated"] is True
        assert output["user_id"] == "u789"
        assert output["has_api_key"] is True
        assert output["has_token"] is True
        assert output["token_valid"] is True
        assert output["backend_url"] == "https://api.test.com"

    def test_status_supports_legacy_token_field(self, k, capsys, tmp_path):
        """Supports 'token' field (legacy) in addition to 'auth_token'."""
        future = (datetime.now(timezone.utc) + timedelta(days=1)).isoformat()
        creds = {
            "user_id": "u-legacy",
            "api_key": "fake-legacy",
            "backend_url": "https://api.test.com",
            "token": "legacy-jwt",  # "token" instead of "auth_token"
            "token_expires": future,
        }
        (tmp_path / "credentials.json").write_text(json.dumps(creds))

        with patch.dict(os.environ, {"KERNLE_DATA_DIR": str(tmp_path)}):
            cmd_auth(_args(auth_action="status", json=True), k)

        output = json.loads(capsys.readouterr().out)
        assert output["has_token"] is True
        assert output["token_valid"] is True

    def test_status_api_key_masking(self, k, capsys, tmp_path):
        """API key is masked in display output."""
        creds = {
            "user_id": "u-mask",
            "api_key": "test-long-key-for-masking-display",
            "backend_url": "https://api.test.com",
        }
        (tmp_path / "credentials.json").write_text(json.dumps(creds))

        with patch.dict(os.environ, {"KERNLE_DATA_DIR": str(tmp_path)}):
            cmd_auth(_args(auth_action="status"), k)

        captured = capsys.readouterr().out
        # Should show masked key with "..."
        assert "..." in captured
        # Full key should NOT appear
        assert "test-long-key-for-masking-display" not in captured

    def test_status_masks_short_api_keys_too(self, k, capsys, tmp_path):
        """Even short keys must never be shown in full."""
        creds = {
            "user_id": "u-mask-short",
            "api_key": "shortkey",
            "backend_url": "https://api.test.com",
        }
        (tmp_path / "credentials.json").write_text(json.dumps(creds))

        with patch.dict(os.environ, {"KERNLE_DATA_DIR": str(tmp_path)}):
            cmd_auth(_args(auth_action="status"), k)

        captured = capsys.readouterr().out
        assert "shortkey" not in captured
        assert "..." in captured


# ============================================================================
# auth register
# ============================================================================


class TestAuthRegister:
    """Tests for `kernle auth register`."""

    def test_register_success(self, k, capsys, tmp_path):
        """Successful registration saves credentials."""
        reg_resp = _make_response(
            201,
            json_data={
                "user_id": "new-user-1",
                "secret": "test-secret",
                "access_token": "jwt-new",
                "expires_in": 604800,
            },
        )
        mock_httpx = MagicMock()
        mock_httpx.post.return_value = reg_resp
        mock_httpx.ConnectError = ConnectionError

        with patch.dict(os.environ, {"KERNLE_DATA_DIR": str(tmp_path)}):
            with patch.dict("sys.modules", {"httpx": mock_httpx}):
                cmd_auth(
                    _args(
                        auth_action="register",
                        backend_url="https://api.test.com",
                        email="test@example.com",
                    ),
                    k,
                )

        captured = capsys.readouterr().out
        assert "Registration successful" in captured
        assert "new-user-1" in captured

        # Verify credentials were saved
        creds_file = tmp_path / "credentials.json"
        assert creds_file.exists()
        saved = json.loads(creds_file.read_text())
        assert saved["user_id"] == "new-user-1"
        assert saved["api_key"] == "test-secret"
        assert saved["backend_url"] == "https://api.test.com"

    def test_register_masks_secret_in_output(self, k, capsys, tmp_path):
        """Registration output should always mask secret values."""
        reg_resp = _make_response(
            201,
            json_data={
                "user_id": "new-user-mask",
                "secret": "shortsecret",
                "access_token": "jwt-new",
                "expires_in": 604800,
            },
        )
        mock_httpx = MagicMock()
        mock_httpx.post.return_value = reg_resp
        mock_httpx.ConnectError = ConnectionError

        with patch.dict(os.environ, {"KERNLE_DATA_DIR": str(tmp_path)}):
            with patch.dict("sys.modules", {"httpx": mock_httpx}):
                cmd_auth(
                    _args(
                        auth_action="register",
                        backend_url="https://api.test.com",
                        email="mask@example.com",
                    ),
                    k,
                )

        captured = capsys.readouterr().out
        assert "shortsecret" not in captured
        assert "Secret:" in captured
        assert "..." in captured

    def test_register_success_json(self, k, capsys, tmp_path):
        """Register --json returns structured output."""
        reg_resp = _make_response(
            200,
            json_data={
                "user_id": "u-json",
                "secret": "test-json-secret",
                "access_token": "jwt",
                "expires_in": 3600,
            },
        )
        mock_httpx = MagicMock()
        mock_httpx.post.return_value = reg_resp
        mock_httpx.ConnectError = ConnectionError

        with patch.dict(os.environ, {"KERNLE_DATA_DIR": str(tmp_path)}):
            with patch.dict("sys.modules", {"httpx": mock_httpx}):
                cmd_auth(
                    _args(
                        auth_action="register",
                        backend_url="https://api.test.com",
                        email="json@test.com",
                        json=True,
                    ),
                    k,
                )

        output = _extract_json(capsys.readouterr().out)
        assert output["status"] == "success"
        assert output["user_id"] == "u-json"

    def test_register_email_conflict(self, k, tmp_path):
        """409 when email already registered."""
        resp = _make_response(409, text="Conflict")
        mock_httpx = MagicMock()
        mock_httpx.post.return_value = resp
        mock_httpx.ConnectError = ConnectionError

        with patch.dict(os.environ, {"KERNLE_DATA_DIR": str(tmp_path)}):
            with patch.dict("sys.modules", {"httpx": mock_httpx}):
                with pytest.raises(SystemExit) as exc_info:
                    cmd_auth(
                        _args(
                            auth_action="register",
                            backend_url="https://api.test.com",
                            email="existing@test.com",
                        ),
                        k,
                    )
                assert exc_info.value.code == 1

    def test_register_bad_request(self, k, tmp_path):
        """400 when request is invalid."""
        resp = _make_response(
            400,
            json_data={"detail": "Invalid email format"},
            text='{"detail":"Invalid email format"}',
        )
        mock_httpx = MagicMock()
        mock_httpx.post.return_value = resp
        mock_httpx.ConnectError = ConnectionError

        with patch.dict(os.environ, {"KERNLE_DATA_DIR": str(tmp_path)}):
            with patch.dict("sys.modules", {"httpx": mock_httpx}):
                with pytest.raises(SystemExit) as exc_info:
                    cmd_auth(
                        _args(
                            auth_action="register",
                            backend_url="https://api.test.com",
                            email="bad-email",
                        ),
                        k,
                    )
                assert exc_info.value.code == 1

    def test_register_server_error(self, k, tmp_path):
        """Exits on 500 server error."""
        resp = _make_response(500, text="Internal Server Error")
        mock_httpx = MagicMock()
        mock_httpx.post.return_value = resp
        mock_httpx.ConnectError = ConnectionError

        with patch.dict(os.environ, {"KERNLE_DATA_DIR": str(tmp_path)}):
            with patch.dict("sys.modules", {"httpx": mock_httpx}):
                with pytest.raises(SystemExit) as exc_info:
                    cmd_auth(
                        _args(
                            auth_action="register",
                            backend_url="https://api.test.com",
                            email="test@test.com",
                        ),
                        k,
                    )
                assert exc_info.value.code == 1

    def test_register_connect_error(self, k, tmp_path):
        """Exits on connection error."""
        mock_httpx = MagicMock()
        mock_httpx.post.side_effect = ConnectionError("Connection refused")
        mock_httpx.ConnectError = ConnectionError

        with patch.dict(os.environ, {"KERNLE_DATA_DIR": str(tmp_path)}):
            with patch.dict("sys.modules", {"httpx": mock_httpx}):
                with pytest.raises(SystemExit) as exc_info:
                    cmd_auth(
                        _args(
                            auth_action="register",
                            backend_url="https://api.test.com",
                            email="test@test.com",
                        ),
                        k,
                    )
                assert exc_info.value.code == 1

    def test_register_prompts_for_email(self, k, capsys, tmp_path):
        """Prompts for email when not provided via args."""
        reg_resp = _make_response(
            201,
            json_data={
                "user_id": "prompted-user",
                "secret": "test-prompted-secret",
                "access_token": "jwt",
                "expires_in": 3600,
            },
        )
        mock_httpx = MagicMock()
        mock_httpx.post.return_value = reg_resp
        mock_httpx.ConnectError = ConnectionError

        with patch.dict(os.environ, {"KERNLE_DATA_DIR": str(tmp_path)}):
            with patch.dict("sys.modules", {"httpx": mock_httpx}):
                with patch("builtins.input", return_value="prompted@test.com"):
                    cmd_auth(
                        _args(
                            auth_action="register",
                            backend_url="https://api.test.com",
                            email=None,
                        ),
                        k,
                    )

        captured = capsys.readouterr().out
        assert "Registration successful" in captured

    def test_register_empty_email_exits(self, k, tmp_path):
        """Exits when empty email entered at prompt."""
        mock_httpx = MagicMock()
        mock_httpx.ConnectError = ConnectionError

        with patch.dict(os.environ, {"KERNLE_DATA_DIR": str(tmp_path)}):
            with patch.dict("sys.modules", {"httpx": mock_httpx}):
                with patch("builtins.input", return_value=""):
                    with pytest.raises(SystemExit) as exc_info:
                        cmd_auth(
                            _args(
                                auth_action="register",
                                backend_url="https://api.test.com",
                                email=None,
                            ),
                            k,
                        )
                    assert exc_info.value.code == 1

    def test_register_invalid_response(self, k, tmp_path):
        """Exits when server response is missing required fields."""
        resp = _make_response(200, json_data={"user_id": None, "secret": None})
        mock_httpx = MagicMock()
        mock_httpx.post.return_value = resp
        mock_httpx.ConnectError = ConnectionError

        with patch.dict(os.environ, {"KERNLE_DATA_DIR": str(tmp_path)}):
            with patch.dict("sys.modules", {"httpx": mock_httpx}):
                with pytest.raises(SystemExit) as exc_info:
                    cmd_auth(
                        _args(
                            auth_action="register",
                            backend_url="https://api.test.com",
                            email="test@test.com",
                        ),
                        k,
                    )
                assert exc_info.value.code == 1


# ============================================================================
# auth login
# ============================================================================


class TestAuthLogin:
    """Tests for `kernle auth login`."""

    def test_login_success(self, k, capsys, tmp_path):
        """Successful login saves credentials."""
        login_resp = _make_response(
            200,
            json_data={
                "user_id": "login-user",
                "token": "jwt-login-token",
                "token_expires": "2025-12-31T23:59:59+00:00",
            },
        )
        mock_httpx = MagicMock()
        mock_httpx.post.return_value = login_resp
        mock_httpx.ConnectError = ConnectionError

        with patch.dict(os.environ, {"KERNLE_DATA_DIR": str(tmp_path)}):
            with patch.dict("sys.modules", {"httpx": mock_httpx}):
                cmd_auth(
                    _args(
                        auth_action="login",
                        backend_url="https://api.test.com",
                        api_key="fake-existing-key",
                    ),
                    k,
                )

        captured = capsys.readouterr().out
        assert "Login successful" in captured

        # Verify credentials saved
        saved = json.loads((tmp_path / "credentials.json").read_text())
        assert saved["user_id"] == "login-user"
        assert saved["auth_token"] == "jwt-login-token"
        assert saved["api_key"] == "fake-existing-key"
        assert saved["backend_url"] == "https://api.test.com"

    def test_login_json_output(self, k, capsys, tmp_path):
        """Login --json returns structured output."""
        login_resp = _make_response(
            200,
            json_data={
                "user_id": "u-json-login",
                "token": "jwt",
                "token_expires": "2025-06-01T00:00:00+00:00",
            },
        )
        mock_httpx = MagicMock()
        mock_httpx.post.return_value = login_resp
        mock_httpx.ConnectError = ConnectionError

        with patch.dict(os.environ, {"KERNLE_DATA_DIR": str(tmp_path)}):
            with patch.dict("sys.modules", {"httpx": mock_httpx}):
                cmd_auth(
                    _args(
                        auth_action="login",
                        backend_url="https://api.test.com",
                        api_key="fake-key",
                        json=True,
                    ),
                    k,
                )

        output = _extract_json(capsys.readouterr().out)
        assert output["status"] == "success"
        assert output["user_id"] == "u-json-login"

    def test_login_invalid_api_key(self, k, tmp_path):
        """401 when API key is invalid."""
        resp = _make_response(401, text="Unauthorized")
        mock_httpx = MagicMock()
        mock_httpx.post.return_value = resp
        mock_httpx.ConnectError = ConnectionError

        with patch.dict(os.environ, {"KERNLE_DATA_DIR": str(tmp_path)}):
            with patch.dict("sys.modules", {"httpx": mock_httpx}):
                with pytest.raises(SystemExit) as exc_info:
                    cmd_auth(
                        _args(
                            auth_action="login",
                            backend_url="https://api.test.com",
                            api_key="bad-key",
                        ),
                        k,
                    )
                assert exc_info.value.code == 1

    def test_login_server_error(self, k, tmp_path):
        """Exits on 500 server error."""
        resp = _make_response(500, text="Server Error")
        mock_httpx = MagicMock()
        mock_httpx.post.return_value = resp
        mock_httpx.ConnectError = ConnectionError

        with patch.dict(os.environ, {"KERNLE_DATA_DIR": str(tmp_path)}):
            with patch.dict("sys.modules", {"httpx": mock_httpx}):
                with pytest.raises(SystemExit) as exc_info:
                    cmd_auth(
                        _args(
                            auth_action="login",
                            backend_url="https://api.test.com",
                            api_key="key",
                        ),
                        k,
                    )
                assert exc_info.value.code == 1

    def test_login_connect_error(self, k, tmp_path):
        """Exits on connection error."""
        mock_httpx = MagicMock()
        mock_httpx.post.side_effect = ConnectionError("refused")
        mock_httpx.ConnectError = ConnectionError

        with patch.dict(os.environ, {"KERNLE_DATA_DIR": str(tmp_path)}):
            with patch.dict("sys.modules", {"httpx": mock_httpx}):
                with pytest.raises(SystemExit) as exc_info:
                    cmd_auth(
                        _args(
                            auth_action="login",
                            backend_url="https://api.test.com",
                            api_key="key",
                        ),
                        k,
                    )
                assert exc_info.value.code == 1

    def test_login_prompts_for_api_key(self, k, capsys, tmp_path):
        """Prompts for API key when not provided."""
        login_resp = _make_response(
            200,
            json_data={
                "user_id": "prompted-login",
                "token": "jwt",
                "token_expires": "2025-12-31T00:00:00+00:00",
            },
        )
        mock_httpx = MagicMock()
        mock_httpx.post.return_value = login_resp
        mock_httpx.ConnectError = ConnectionError

        with patch.dict(os.environ, {"KERNLE_DATA_DIR": str(tmp_path)}):
            with patch.dict("sys.modules", {"httpx": mock_httpx}):
                with patch("getpass.getpass", return_value="fake-prompted-key"):
                    cmd_auth(
                        _args(
                            auth_action="login",
                            backend_url="https://api.test.com",
                            api_key=None,
                        ),
                        k,
                    )

        captured = capsys.readouterr().out
        assert "Login successful" in captured

    def test_login_uses_existing_credentials(self, k, capsys, tmp_path):
        """Uses backend_url and api_key from existing credentials."""
        existing = {
            "backend_url": "https://saved.api.com",
            "api_key": "fake-saved-key",
        }
        (tmp_path / "credentials.json").write_text(json.dumps(existing))

        login_resp = _make_response(
            200,
            json_data={
                "user_id": "saved-user",
                "token": "jwt-saved",
                "token_expires": "2025-12-31T00:00:00+00:00",
            },
        )
        mock_httpx = MagicMock()
        mock_httpx.post.return_value = login_resp
        mock_httpx.ConnectError = ConnectionError

        with patch.dict(os.environ, {"KERNLE_DATA_DIR": str(tmp_path)}):
            with patch.dict("sys.modules", {"httpx": mock_httpx}):
                cmd_auth(
                    _args(auth_action="login", backend_url=None, api_key=None),
                    k,
                )

        captured = capsys.readouterr().out
        assert "Login successful" in captured

        # Verify the saved backend URL was used
        call_args = mock_httpx.post.call_args
        assert "saved.api.com" in call_args[0][0]


# ============================================================================
# auth logout
# ============================================================================


class TestAuthLogout:
    """Tests for `kernle auth logout`."""

    def test_logout_removes_credentials(self, k, capsys, tmp_path):
        """Logout removes credential file."""
        (tmp_path / "credentials.json").write_text(json.dumps({"user_id": "u1"}))

        with patch.dict(os.environ, {"KERNLE_DATA_DIR": str(tmp_path)}):
            cmd_auth(_args(auth_action="logout"), k)

        captured = capsys.readouterr().out
        assert "Logged out" in captured
        assert not (tmp_path / "credentials.json").exists()

    def test_logout_json(self, k, capsys, tmp_path):
        """Logout --json returns structured output."""
        (tmp_path / "credentials.json").write_text(json.dumps({"user_id": "u1"}))

        with patch.dict(os.environ, {"KERNLE_DATA_DIR": str(tmp_path)}):
            cmd_auth(_args(auth_action="logout", json=True), k)

        output = json.loads(capsys.readouterr().out)
        assert output["status"] == "success"

    def test_logout_when_already_logged_out(self, k, capsys, tmp_path):
        """Shows message when no credentials to clear."""
        with patch.dict(os.environ, {"KERNLE_DATA_DIR": str(tmp_path)}):
            cmd_auth(_args(auth_action="logout"), k)

        captured = capsys.readouterr().out
        assert "Already logged out" in captured

    def test_logout_already_out_json(self, k, capsys, tmp_path):
        """JSON output for already-logged-out state."""
        with patch.dict(os.environ, {"KERNLE_DATA_DIR": str(tmp_path)}):
            cmd_auth(_args(auth_action="logout", json=True), k)

        output = json.loads(capsys.readouterr().out)
        assert output["status"] == "success"


# ============================================================================
# auth keys list
# ============================================================================


class TestAuthKeysList:
    """Tests for `kernle auth keys list`."""

    def test_keys_list_success(self, capsys, tmp_path):
        """Lists API keys from backend."""
        creds = {"backend_url": "https://api.test.com", "api_key": "fake-master"}
        (tmp_path / "credentials.json").write_text(json.dumps(creds))

        keys_data = [
            {
                "id": "key-1",
                "name": "My Key",
                "key_prefix": "test-prefix",
                "created_at": "2025-01-01T00:00:00Z",
                "last_used_at": "2025-06-01T12:00:00Z",
                "is_active": True,
            },
            {
                "id": "key-2",
                "name": "Old Key",
                "key_prefix": "test-old",
                "created_at": "2024-06-01T00:00:00Z",
                "is_active": False,
            },
        ]
        resp = _make_response(200, json_data=keys_data)
        mock_httpx = MagicMock()
        mock_httpx.get.return_value = resp
        mock_httpx.ConnectError = ConnectionError

        with patch.dict(os.environ, {"KERNLE_DATA_DIR": str(tmp_path)}):
            with patch.dict("sys.modules", {"httpx": mock_httpx}):
                cmd_auth_keys(_args(auth_action="keys", keys_action="list"))

        captured = capsys.readouterr().out
        assert "My Key" in captured
        assert "Old Key" in captured
        assert "REVOKED" in captured

    def test_keys_list_json(self, capsys, tmp_path):
        """List --json returns raw backend response."""
        creds = {"backend_url": "https://api.test.com", "api_key": "fake-master"}
        (tmp_path / "credentials.json").write_text(json.dumps(creds))

        keys_data = [{"id": "k1", "name": "Test", "key_prefix": "fake-test"}]
        resp = _make_response(200, json_data=keys_data)
        mock_httpx = MagicMock()
        mock_httpx.get.return_value = resp
        mock_httpx.ConnectError = ConnectionError

        with patch.dict(os.environ, {"KERNLE_DATA_DIR": str(tmp_path)}):
            with patch.dict("sys.modules", {"httpx": mock_httpx}):
                cmd_auth_keys(_args(auth_action="keys", keys_action="list", json=True))

        output = json.loads(capsys.readouterr().out)
        assert len(output) == 1
        assert output[0]["id"] == "k1"

    def test_keys_list_empty(self, capsys, tmp_path):
        """Shows message when no keys found."""
        creds = {"backend_url": "https://api.test.com", "api_key": "fake-master"}
        (tmp_path / "credentials.json").write_text(json.dumps(creds))

        resp = _make_response(200, json_data=[])
        mock_httpx = MagicMock()
        mock_httpx.get.return_value = resp
        mock_httpx.ConnectError = ConnectionError

        with patch.dict(os.environ, {"KERNLE_DATA_DIR": str(tmp_path)}):
            with patch.dict("sys.modules", {"httpx": mock_httpx}):
                cmd_auth_keys(_args(auth_action="keys", keys_action="list"))

        captured = capsys.readouterr().out
        assert "No API keys found" in captured

    def test_keys_list_auth_failure(self, tmp_path):
        """Exits on 401."""
        creds = {"backend_url": "https://api.test.com", "api_key": "fake-bad"}
        (tmp_path / "credentials.json").write_text(json.dumps(creds))

        resp = _make_response(401, text="Unauthorized")
        mock_httpx = MagicMock()
        mock_httpx.get.return_value = resp
        mock_httpx.ConnectError = ConnectionError

        with patch.dict(os.environ, {"KERNLE_DATA_DIR": str(tmp_path)}):
            with patch.dict("sys.modules", {"httpx": mock_httpx}):
                with pytest.raises(SystemExit) as exc_info:
                    cmd_auth_keys(_args(auth_action="keys", keys_action="list"))
                assert exc_info.value.code == 1

    def test_keys_list_no_credentials(self, tmp_path):
        """Exits when no credentials configured."""
        with patch.dict(os.environ, {"KERNLE_DATA_DIR": str(tmp_path)}):
            with pytest.raises(SystemExit) as exc_info:
                mock_httpx = MagicMock()
                mock_httpx.ConnectError = ConnectionError
                with patch.dict("sys.modules", {"httpx": mock_httpx}):
                    cmd_auth_keys(_args(auth_action="keys", keys_action="list"))
            assert exc_info.value.code == 1


# ============================================================================
# auth keys create
# ============================================================================


class TestAuthKeysCreate:
    """Tests for `kernle auth keys create`."""

    def test_keys_create_success(self, capsys, tmp_path):
        """Creates a new API key."""
        creds = {"backend_url": "https://api.test.com", "api_key": "fake-master"}
        (tmp_path / "credentials.json").write_text(json.dumps(creds))

        resp = _make_response(
            201,
            json_data={
                "id": "new-key-id",
                "name": "CI/CD Key",
                "key": "test-new-key",
            },
        )
        mock_httpx = MagicMock()
        mock_httpx.post.return_value = resp
        mock_httpx.ConnectError = ConnectionError

        with patch.dict(os.environ, {"KERNLE_DATA_DIR": str(tmp_path)}):
            with patch.dict("sys.modules", {"httpx": mock_httpx}):
                cmd_auth_keys(_args(auth_action="keys", keys_action="create", name="CI/CD Key"))

        captured = capsys.readouterr().out
        assert "API key created" in captured
        assert "test-new-key" in captured
        assert "SAVE THIS KEY" in captured

    def test_keys_create_json(self, capsys, tmp_path):
        """Create --json returns only allowlisted fields."""
        creds = {"backend_url": "https://api.test.com", "api_key": "fake-master"}
        (tmp_path / "credentials.json").write_text(json.dumps(creds))

        resp = _make_response(
            200,
            json_data={
                "id": "k-new",
                "key": "fake-newkey",
                "name": "Test",
                "access_token": "should-not-leak",
                "secret": "should-not-leak",
                "refresh_token": "should-not-leak",
            },
        )
        mock_httpx = MagicMock()
        mock_httpx.post.return_value = resp
        mock_httpx.ConnectError = ConnectionError

        with patch.dict(os.environ, {"KERNLE_DATA_DIR": str(tmp_path)}):
            with patch.dict("sys.modules", {"httpx": mock_httpx}):
                cmd_auth_keys(
                    _args(auth_action="keys", keys_action="create", name="Test", json=True)
                )

        output = json.loads(capsys.readouterr().out)
        assert output == {"id": "k-new", "name": "Test", "key": "fake-newkey"}
        assert "access_token" not in output
        assert "secret" not in output
        assert "refresh_token" not in output

    def test_keys_create_rate_limited(self, tmp_path):
        """429 rate limit exits."""
        creds = {"backend_url": "https://api.test.com", "api_key": "fake-master"}
        (tmp_path / "credentials.json").write_text(json.dumps(creds))

        resp = _make_response(429, text="Too Many Requests")
        mock_httpx = MagicMock()
        mock_httpx.post.return_value = resp
        mock_httpx.ConnectError = ConnectionError

        with patch.dict(os.environ, {"KERNLE_DATA_DIR": str(tmp_path)}):
            with patch.dict("sys.modules", {"httpx": mock_httpx}):
                with pytest.raises(SystemExit) as exc_info:
                    cmd_auth_keys(_args(auth_action="keys", keys_action="create", name="X"))
                assert exc_info.value.code == 1

    def test_keys_create_auth_failure(self, tmp_path):
        """401 exits with auth error."""
        creds = {"backend_url": "https://api.test.com", "api_key": "fake-bad"}
        (tmp_path / "credentials.json").write_text(json.dumps(creds))

        resp = _make_response(401, text="Unauthorized")
        mock_httpx = MagicMock()
        mock_httpx.post.return_value = resp
        mock_httpx.ConnectError = ConnectionError

        with patch.dict(os.environ, {"KERNLE_DATA_DIR": str(tmp_path)}):
            with patch.dict("sys.modules", {"httpx": mock_httpx}):
                with pytest.raises(SystemExit) as exc_info:
                    cmd_auth_keys(_args(auth_action="keys", keys_action="create", name="X"))
                assert exc_info.value.code == 1


# ============================================================================
# auth keys revoke
# ============================================================================


class TestAuthKeysRevoke:
    """Tests for `kernle auth keys revoke`."""

    def test_keys_revoke_success(self, capsys, tmp_path):
        """Revokes an API key with --force."""
        creds = {"backend_url": "https://api.test.com", "api_key": "fake-master"}
        (tmp_path / "credentials.json").write_text(json.dumps(creds))

        resp = _make_response(200)
        mock_httpx = MagicMock()
        mock_httpx.delete.return_value = resp
        mock_httpx.ConnectError = ConnectionError

        with patch.dict(os.environ, {"KERNLE_DATA_DIR": str(tmp_path)}):
            with patch.dict("sys.modules", {"httpx": mock_httpx}):
                cmd_auth_keys(
                    _args(
                        auth_action="keys", keys_action="revoke", key_id="key-to-revoke", force=True
                    )
                )

        captured = capsys.readouterr().out
        assert "revoked" in captured
        assert "key-to-revoke" in captured

    def test_keys_revoke_json(self, capsys, tmp_path):
        """Revoke --json returns structured output."""
        creds = {"backend_url": "https://api.test.com", "api_key": "fake-master"}
        (tmp_path / "credentials.json").write_text(json.dumps(creds))

        resp = _make_response(204)
        mock_httpx = MagicMock()
        mock_httpx.delete.return_value = resp
        mock_httpx.ConnectError = ConnectionError

        with patch.dict(os.environ, {"KERNLE_DATA_DIR": str(tmp_path)}):
            with patch.dict("sys.modules", {"httpx": mock_httpx}):
                cmd_auth_keys(
                    _args(
                        auth_action="keys",
                        keys_action="revoke",
                        key_id="k-del",
                        force=True,
                        json=True,
                    )
                )

        output = json.loads(capsys.readouterr().out)
        assert output["status"] == "success"
        assert output["key_id"] == "k-del"
        assert output["action"] == "revoked"

    def test_keys_revoke_not_found(self, tmp_path):
        """404 when key not found."""
        creds = {"backend_url": "https://api.test.com", "api_key": "fake-master"}
        (tmp_path / "credentials.json").write_text(json.dumps(creds))

        resp = _make_response(404, text="Not Found")
        mock_httpx = MagicMock()
        mock_httpx.delete.return_value = resp
        mock_httpx.ConnectError = ConnectionError

        with patch.dict(os.environ, {"KERNLE_DATA_DIR": str(tmp_path)}):
            with patch.dict("sys.modules", {"httpx": mock_httpx}):
                with pytest.raises(SystemExit) as exc_info:
                    cmd_auth_keys(
                        _args(
                            auth_action="keys",
                            keys_action="revoke",
                            key_id="nonexistent",
                            force=True,
                        )
                    )
                assert exc_info.value.code == 1

    def test_keys_revoke_with_confirmation(self, capsys, tmp_path):
        """Prompts for confirmation without --force."""
        creds = {"backend_url": "https://api.test.com", "api_key": "fake-master"}
        (tmp_path / "credentials.json").write_text(json.dumps(creds))

        resp = _make_response(200)
        mock_httpx = MagicMock()
        mock_httpx.delete.return_value = resp
        mock_httpx.ConnectError = ConnectionError

        with patch.dict(os.environ, {"KERNLE_DATA_DIR": str(tmp_path)}):
            with patch.dict("sys.modules", {"httpx": mock_httpx}):
                with patch("builtins.input", return_value="yes"):
                    cmd_auth_keys(
                        _args(
                            auth_action="keys",
                            keys_action="revoke",
                            key_id="key-confirm",
                            force=False,
                        )
                    )

        captured = capsys.readouterr().out
        assert "revoked" in captured

    def test_keys_revoke_aborted(self, capsys, tmp_path):
        """Aborts when user doesn't confirm."""
        creds = {"backend_url": "https://api.test.com", "api_key": "fake-master"}
        (tmp_path / "credentials.json").write_text(json.dumps(creds))

        mock_httpx = MagicMock()
        mock_httpx.ConnectError = ConnectionError

        with patch.dict(os.environ, {"KERNLE_DATA_DIR": str(tmp_path)}):
            with patch.dict("sys.modules", {"httpx": mock_httpx}):
                with patch("builtins.input", return_value="no"):
                    cmd_auth_keys(
                        _args(
                            auth_action="keys",
                            keys_action="revoke",
                            key_id="key-nope",
                            force=False,
                        )
                    )

        captured = capsys.readouterr().out
        assert "Aborted" in captured
        # Should NOT have made a delete request
        mock_httpx.delete.assert_not_called()


# ============================================================================
# auth keys cycle
# ============================================================================


class TestAuthKeysCycle:
    """Tests for `kernle auth keys cycle`."""

    def test_keys_cycle_success(self, capsys, tmp_path):
        """Cycles an API key with --force."""
        creds = {"backend_url": "https://api.test.com", "api_key": "fake-master"}
        (tmp_path / "credentials.json").write_text(json.dumps(creds))

        resp = _make_response(
            200,
            json_data={
                "id": "new-key-id",
                "name": "Cycled Key",
                "key": "fake-cycled-new-key-12345",
            },
        )
        mock_httpx = MagicMock()
        mock_httpx.post.return_value = resp
        mock_httpx.ConnectError = ConnectionError

        with patch.dict(os.environ, {"KERNLE_DATA_DIR": str(tmp_path)}):
            with patch.dict("sys.modules", {"httpx": mock_httpx}):
                cmd_auth_keys(
                    _args(
                        auth_action="keys",
                        keys_action="cycle",
                        key_id="old-key-id",
                        force=True,
                    )
                )

        captured = capsys.readouterr().out
        assert "API key cycled" in captured
        assert "fake-cycled-new-key-12345" in captured
        assert "SAVE THIS NEW KEY" in captured

    def test_keys_cycle_json(self, capsys, tmp_path):
        """Cycle --json returns only allowlisted fields."""
        creds = {"backend_url": "https://api.test.com", "api_key": "fake-master"}
        (tmp_path / "credentials.json").write_text(json.dumps(creds))

        resp = _make_response(
            201,
            json_data={
                "key_id": "new-id",
                "api_key": "fake-new-cycled",
                "name": "Key",
                "jwt": "should-not-leak",
                "secret": "should-not-leak",
            },
        )
        mock_httpx = MagicMock()
        mock_httpx.post.return_value = resp
        mock_httpx.ConnectError = ConnectionError

        with patch.dict(os.environ, {"KERNLE_DATA_DIR": str(tmp_path)}):
            with patch.dict("sys.modules", {"httpx": mock_httpx}):
                cmd_auth_keys(
                    _args(
                        auth_action="keys",
                        keys_action="cycle",
                        key_id="old-id",
                        force=True,
                        json=True,
                    )
                )

        output = json.loads(capsys.readouterr().out)
        assert output == {
            "id": "new-id",
            "name": "Key",
            "key": "fake-new-cycled",
            "old_key_id": "old-id",
        }
        assert "jwt" not in output
        assert "secret" not in output

    def test_keys_cycle_not_found(self, tmp_path):
        """404 when key not found."""
        creds = {"backend_url": "https://api.test.com", "api_key": "fake-master"}
        (tmp_path / "credentials.json").write_text(json.dumps(creds))

        resp = _make_response(404, text="Not Found")
        mock_httpx = MagicMock()
        mock_httpx.post.return_value = resp
        mock_httpx.ConnectError = ConnectionError

        with patch.dict(os.environ, {"KERNLE_DATA_DIR": str(tmp_path)}):
            with patch.dict("sys.modules", {"httpx": mock_httpx}):
                with pytest.raises(SystemExit) as exc_info:
                    cmd_auth_keys(
                        _args(
                            auth_action="keys",
                            keys_action="cycle",
                            key_id="gone",
                            force=True,
                        )
                    )
                assert exc_info.value.code == 1

    def test_keys_cycle_with_confirmation(self, capsys, tmp_path):
        """Prompts for confirmation without --force."""
        creds = {"backend_url": "https://api.test.com", "api_key": "fake-master"}
        (tmp_path / "credentials.json").write_text(json.dumps(creds))

        resp = _make_response(
            200,
            json_data={"id": "cycled", "key": "fake-c", "name": "K"},
        )
        mock_httpx = MagicMock()
        mock_httpx.post.return_value = resp
        mock_httpx.ConnectError = ConnectionError

        with patch.dict(os.environ, {"KERNLE_DATA_DIR": str(tmp_path)}):
            with patch.dict("sys.modules", {"httpx": mock_httpx}):
                with patch("builtins.input", return_value="yes"):
                    cmd_auth_keys(
                        _args(
                            auth_action="keys",
                            keys_action="cycle",
                            key_id="k-cycle",
                            force=False,
                        )
                    )

        captured = capsys.readouterr().out
        assert "API key cycled" in captured

    def test_keys_cycle_aborted(self, capsys, tmp_path):
        """Aborts when user doesn't confirm."""
        creds = {"backend_url": "https://api.test.com", "api_key": "fake-master"}
        (tmp_path / "credentials.json").write_text(json.dumps(creds))

        mock_httpx = MagicMock()
        mock_httpx.ConnectError = ConnectionError

        with patch.dict(os.environ, {"KERNLE_DATA_DIR": str(tmp_path)}):
            with patch.dict("sys.modules", {"httpx": mock_httpx}):
                with patch("builtins.input", return_value="no"):
                    cmd_auth_keys(
                        _args(
                            auth_action="keys",
                            keys_action="cycle",
                            key_id="k-nope",
                            force=False,
                        )
                    )

        captured = capsys.readouterr().out
        assert "Aborted" in captured
        # Should NOT have made a post request for cycle
        mock_httpx.post.assert_not_called()

    def test_keys_cycle_connect_error(self, tmp_path):
        """Exits on connection error."""
        creds = {"backend_url": "https://api.test.com", "api_key": "fake-master"}
        (tmp_path / "credentials.json").write_text(json.dumps(creds))

        mock_httpx = MagicMock()
        mock_httpx.post.side_effect = ConnectionError("refused")
        mock_httpx.ConnectError = ConnectionError

        with patch.dict(os.environ, {"KERNLE_DATA_DIR": str(tmp_path)}):
            with patch.dict("sys.modules", {"httpx": mock_httpx}):
                with pytest.raises(SystemExit) as exc_info:
                    cmd_auth_keys(
                        _args(
                            auth_action="keys",
                            keys_action="cycle",
                            key_id="k-err",
                            force=True,
                        )
                    )
                assert exc_info.value.code == 1
