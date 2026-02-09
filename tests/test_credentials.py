"""Tests for kernle.cli.commands.credentials module."""

import json
import os
from pathlib import Path
from unittest.mock import patch

import pytest

from kernle.cli.commands.credentials import (
    clear_credentials,
    get_credentials_path,
    load_credentials,
    prompt_backend_url,
    save_credentials,
    warn_non_https_url,
)

# ============================================================================
# get_credentials_path
# ============================================================================


class TestGetCredentialsPath:
    """Tests for get_credentials_path()."""

    def test_default_path(self):
        """Uses ~/.kernle/credentials.json when no env var is set."""
        with patch.dict(os.environ, {}, clear=False):
            # Remove KERNLE_DATA_DIR if it exists
            env = os.environ.copy()
            env.pop("KERNLE_DATA_DIR", None)
            with patch.dict(os.environ, env, clear=True):
                path = get_credentials_path()
                assert path == Path.home() / ".kernle" / "credentials.json"

    def test_custom_data_dir(self, tmp_path):
        """Respects KERNLE_DATA_DIR environment variable."""
        with patch.dict(os.environ, {"KERNLE_DATA_DIR": str(tmp_path)}):
            path = get_credentials_path()
            assert path == tmp_path / "credentials.json"


# ============================================================================
# load_credentials
# ============================================================================


class TestLoadCredentials:
    """Tests for load_credentials()."""

    def test_returns_none_when_file_missing(self, tmp_path):
        """Returns None when credentials file does not exist."""
        with patch.dict(os.environ, {"KERNLE_DATA_DIR": str(tmp_path)}):
            result = load_credentials()
            assert result is None

    def test_loads_valid_json(self, tmp_path):
        """Loads and returns valid JSON credentials."""
        creds = {"user_id": "u123", "api_key": "secret", "backend_url": "https://api.example.com"}
        creds_file = tmp_path / "credentials.json"
        creds_file.write_text(json.dumps(creds))

        with patch.dict(os.environ, {"KERNLE_DATA_DIR": str(tmp_path)}):
            result = load_credentials()
            assert result == creds

    def test_returns_none_for_corrupt_json(self, tmp_path):
        """Returns None for invalid JSON content."""
        creds_file = tmp_path / "credentials.json"
        creds_file.write_text("{not valid json!!!")

        with patch.dict(os.environ, {"KERNLE_DATA_DIR": str(tmp_path)}):
            result = load_credentials()
            assert result is None

    def test_returns_none_for_empty_file(self, tmp_path):
        """Returns None for empty file (JSONDecodeError)."""
        creds_file = tmp_path / "credentials.json"
        creds_file.write_text("")

        with patch.dict(os.environ, {"KERNLE_DATA_DIR": str(tmp_path)}):
            result = load_credentials()
            assert result is None


# ============================================================================
# save_credentials
# ============================================================================


class TestSaveCredentials:
    """Tests for save_credentials()."""

    def test_saves_json_to_file(self, tmp_path):
        """Saves credentials dict as JSON to the credentials file."""
        creds = {"user_id": "u999", "backend_url": "https://api.test.com"}

        with patch.dict(os.environ, {"KERNLE_DATA_DIR": str(tmp_path)}):
            save_credentials(creds)

            # Verify file contents
            creds_file = tmp_path / "credentials.json"
            assert creds_file.exists()
            loaded = json.loads(creds_file.read_text())
            assert loaded == creds

    def test_sets_restrictive_permissions(self, tmp_path):
        """Sets file permissions to 0o600 (owner read/write only)."""
        with patch.dict(os.environ, {"KERNLE_DATA_DIR": str(tmp_path)}):
            save_credentials({"key": "value"})

            creds_file = tmp_path / "credentials.json"
            mode = creds_file.stat().st_mode & 0o777
            assert mode == 0o600

    def test_creates_parent_directory(self, tmp_path):
        """Creates parent directory if it doesn't exist."""
        nested = tmp_path / "deep" / "nested"

        with patch.dict(os.environ, {"KERNLE_DATA_DIR": str(nested)}):
            save_credentials({"user_id": "u1"})

            creds_file = nested / "credentials.json"
            assert creds_file.exists()
            assert json.loads(creds_file.read_text())["user_id"] == "u1"

    def test_overwrites_existing_file(self, tmp_path):
        """Overwrites existing credentials file."""
        creds_file = tmp_path / "credentials.json"
        creds_file.write_text(json.dumps({"old": "data"}))

        with patch.dict(os.environ, {"KERNLE_DATA_DIR": str(tmp_path)}):
            save_credentials({"new": "data"})
            loaded = json.loads(creds_file.read_text())
            assert loaded == {"new": "data"}
            assert "old" not in loaded


# ============================================================================
# clear_credentials
# ============================================================================


class TestClearCredentials:
    """Tests for clear_credentials()."""

    def test_removes_existing_file(self, tmp_path):
        """Removes credentials file and returns True."""
        creds_file = tmp_path / "credentials.json"
        creds_file.write_text("{}")

        with patch.dict(os.environ, {"KERNLE_DATA_DIR": str(tmp_path)}):
            result = clear_credentials()
            assert result is True
            assert not creds_file.exists()

    def test_returns_false_when_no_file(self, tmp_path):
        """Returns False when credentials file does not exist."""
        with patch.dict(os.environ, {"KERNLE_DATA_DIR": str(tmp_path)}):
            result = clear_credentials()
            assert result is False


# ============================================================================
# prompt_backend_url
# ============================================================================


class TestPromptBackendUrl:
    """Tests for prompt_backend_url()."""

    def test_returns_user_input(self):
        """Returns the URL entered by the user."""
        with patch("builtins.input", return_value="https://custom.api.com"):
            result = prompt_backend_url()
            assert result == "https://custom.api.com"

    def test_returns_default_when_empty_input(self):
        """Returns default URL when user presses enter."""
        with patch("builtins.input", return_value=""):
            result = prompt_backend_url()
            assert result == "https://api.kernle.io"

    def test_returns_current_url_as_default(self):
        """Uses current_url as default when provided."""
        with patch("builtins.input", return_value=""):
            result = prompt_backend_url(current_url="https://existing.api.com")
            assert result == "https://existing.api.com"

    def test_strips_whitespace(self):
        """Strips whitespace from user input."""
        with patch("builtins.input", return_value="  https://api.test.com  "):
            result = prompt_backend_url()
            assert result == "https://api.test.com"

    def test_warns_non_https(self, capsys):
        """Warns when user enters non-HTTPS URL (not localhost)."""
        with patch("builtins.input", return_value="http://production.server.com"):
            result = prompt_backend_url()
            assert result == "http://production.server.com"
            captured = capsys.readouterr()
            assert "WARNING" in captured.out
            assert "cleartext" in captured.out

    def test_no_warn_for_localhost(self, capsys):
        """No warning for http://localhost (development)."""
        with patch("builtins.input", return_value="http://localhost:8000"):
            result = prompt_backend_url()
            assert result == "http://localhost:8000"
            captured = capsys.readouterr()
            assert "WARNING" not in captured.out or "cleartext" not in captured.out

    def test_no_warn_for_127_0_0_1(self, capsys):
        """No warning for http://127.0.0.1 (development)."""
        with patch("builtins.input", return_value="http://127.0.0.1:3000"):
            result = prompt_backend_url()
            assert result == "http://127.0.0.1:3000"
            captured = capsys.readouterr()
            # Should not contain the "cleartext" warning
            assert "cleartext" not in captured.out

    def test_exits_on_eof(self):
        """Exits with code 1 on EOFError."""
        with patch("builtins.input", side_effect=EOFError):
            with pytest.raises(SystemExit) as exc_info:
                prompt_backend_url()
            assert exc_info.value.code == 1

    def test_exits_on_keyboard_interrupt(self):
        """Exits with code 1 on KeyboardInterrupt."""
        with patch("builtins.input", side_effect=KeyboardInterrupt):
            with pytest.raises(SystemExit) as exc_info:
                prompt_backend_url()
            assert exc_info.value.code == 1


# ============================================================================
# warn_non_https_url
# ============================================================================


class TestWarnNonHttpsUrl:
    """Tests for warn_non_https_url()."""

    def test_no_warning_for_https(self, capsys):
        """No output for HTTPS URLs."""
        warn_non_https_url("https://api.kernle.io")
        captured = capsys.readouterr()
        assert captured.out == ""

    def test_no_warning_for_none(self, capsys):
        """No output for None URL."""
        warn_non_https_url(None)
        captured = capsys.readouterr()
        assert captured.out == ""

    def test_no_warning_for_empty_string(self, capsys):
        """No output for empty string."""
        warn_non_https_url("")
        captured = capsys.readouterr()
        assert captured.out == ""

    def test_no_warning_for_localhost(self, capsys):
        """No warning for http://localhost."""
        warn_non_https_url("http://localhost:8000")
        captured = capsys.readouterr()
        assert captured.out == ""

    def test_no_warning_for_127_0_0_1(self, capsys):
        """No warning for http://127.0.0.1."""
        warn_non_https_url("http://127.0.0.1:5000")
        captured = capsys.readouterr()
        assert captured.out == ""

    def test_warns_for_http_production(self, capsys):
        """Prints warning for non-HTTPS, non-localhost URL."""
        warn_non_https_url("http://api.production.com")
        captured = capsys.readouterr()
        assert "WARNING" in captured.out
        assert "cleartext" in captured.out

    def test_includes_source_in_warning(self, capsys):
        """Includes source context in warning message."""
        warn_non_https_url("http://api.example.com", source="env")
        captured = capsys.readouterr()
        assert "(from env)" in captured.out

    def test_no_source_when_none(self, capsys):
        """No source parenthetical when source is None."""
        warn_non_https_url("http://api.example.com")
        captured = capsys.readouterr()
        assert "(from" not in captured.out
