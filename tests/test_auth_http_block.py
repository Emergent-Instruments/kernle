"""Tests for blocking plaintext HTTP credential submission."""

from unittest.mock import MagicMock, patch

import pytest

from kernle.cli.commands.credentials import require_https_url


class TestRequireHttpsUrl:
    """Tests for the require_https_url blocking function."""

    def test_blocks_non_https_url(self):
        """Plain http:// to a remote host must raise SystemExit."""
        with pytest.raises(SystemExit):
            require_https_url("http://evil.com")

    def test_allows_https(self):
        """https:// URLs must pass through without error."""
        require_https_url("https://api.example.com")  # should not raise

    def test_allows_localhost_http(self):
        """http://localhost is allowed for development."""
        require_https_url("http://localhost:8000")  # should not raise

    def test_allows_127_0_0_1_http(self):
        """http://127.0.0.1 is allowed for development."""
        require_https_url("http://127.0.0.1:8000")  # should not raise

    def test_allows_empty_url(self):
        """Empty or None URL passes through (other code handles missing URLs)."""
        require_https_url("")  # should not raise
        require_https_url(None)  # should not raise

    def test_includes_source_in_message(self, capsys):
        """Error message includes the source when provided."""
        with pytest.raises(SystemExit):
            require_https_url("http://evil.com", source="args")
        captured = capsys.readouterr()
        assert "args" in captured.out

    def test_includes_url_in_message(self, capsys):
        """Error message includes the offending URL."""
        with pytest.raises(SystemExit):
            require_https_url("http://evil.com/api")
        captured = capsys.readouterr()
        assert "http://evil.com/api" in captured.out

    def test_blocks_localhost_dot_evil(self):
        """http://localhost.evil.com must be blocked (not real localhost)."""
        with pytest.raises(SystemExit):
            require_https_url("http://localhost.evil.com")

    def test_blocks_localhost_at_evil(self):
        """http://localhost@evil.com must be blocked (userinfo bypass)."""
        with pytest.raises(SystemExit):
            require_https_url("http://localhost@evil.com")

    def test_blocks_127_dot_evil(self):
        """http://127.0.0.1.evil.com must be blocked."""
        with pytest.raises(SystemExit):
            require_https_url("http://127.0.0.1.evil.com")

    def test_allows_localhost_with_port(self):
        """http://localhost:3000 is valid localhost."""
        require_https_url("http://localhost:3000")  # should not raise

    def test_allows_localhost_with_path(self):
        """http://localhost/api/v1 is valid localhost."""
        require_https_url("http://localhost/api/v1")  # should not raise

    def test_allows_bare_localhost(self):
        """http://localhost with no port or path is valid."""
        require_https_url("http://localhost")  # should not raise


def _make_login_args(backend_url="https://api.example.com", api_key=None, json_flag=False):
    """Create a mock args object for the login flow."""
    args = MagicMock()
    args.auth_action = "login"
    args.backend_url = backend_url
    args.api_key = api_key or "test-secret-key-12345"
    args.json = json_flag
    return args


def _make_register_args(backend_url="https://api.example.com", email=None, json_flag=False):
    """Create a mock args object for the register flow."""
    args = MagicMock()
    args.auth_action = "register"
    args.backend_url = backend_url
    args.email = email or "test@example.com"
    args.json = json_flag
    return args


class TestLoginBlocksHttp:
    """Tests that the login flow blocks non-HTTPS URLs."""

    @patch("kernle.cli.commands.auth.load_credentials", return_value=None)
    def test_login_blocks_non_https_url(self, mock_creds):
        """Login with http://evil.com must exit before sending credentials."""
        from kernle.cli.commands.auth import cmd_auth

        args = _make_login_args(backend_url="http://evil.com")

        with pytest.raises(SystemExit):
            cmd_auth(args)

    @patch("kernle.cli.commands.auth.save_credentials")
    @patch("kernle.cli.commands.auth.load_credentials", return_value=None)
    def test_login_allows_https(self, mock_creds, mock_save):
        """Login with https:// should proceed to send the request."""
        from kernle.cli.commands.auth import cmd_auth

        # Mock httpx at the module level so get_http_client() returns our mock
        mock_httpx = MagicMock()
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "user_id": "u123",
            "token": "tok",
            "token_expires": "2099-01-01T00:00:00+00:00",
        }
        mock_httpx.post.return_value = mock_response
        mock_httpx.ConnectError = Exception

        args = _make_login_args(backend_url="https://api.example.com")

        with patch.dict("sys.modules", {"httpx": mock_httpx}):
            cmd_auth(args)

        # Verify that httpx.post was called (credentials were sent over HTTPS)
        mock_httpx.post.assert_called_once()

    @patch("kernle.cli.commands.auth.save_credentials")
    @patch("kernle.cli.commands.auth.load_credentials", return_value=None)
    def test_login_allows_localhost_http(self, mock_creds, mock_save):
        """Login with http://localhost should proceed (dev exception)."""
        from kernle.cli.commands.auth import cmd_auth

        mock_httpx = MagicMock()
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "user_id": "u123",
            "token": "tok",
            "token_expires": "2099-01-01T00:00:00+00:00",
        }
        mock_httpx.post.return_value = mock_response
        mock_httpx.ConnectError = Exception

        args = _make_login_args(backend_url="http://localhost:8000")

        with patch.dict("sys.modules", {"httpx": mock_httpx}):
            cmd_auth(args)

        mock_httpx.post.assert_called_once()


class TestRegisterBlocksHttp:
    """Tests that the register flow blocks non-HTTPS URLs."""

    @patch("kernle.cli.commands.auth.load_credentials", return_value=None)
    def test_register_blocks_non_https_url(self, mock_creds):
        """Register with http://evil.com must exit before sending credentials."""
        from kernle.cli.commands.auth import cmd_auth

        args = _make_register_args(backend_url="http://evil.com")

        with pytest.raises(SystemExit):
            cmd_auth(args)

    @patch("kernle.cli.commands.auth.save_credentials")
    @patch("kernle.cli.commands.auth.load_credentials", return_value=None)
    def test_register_allows_https(self, mock_creds, mock_save):
        """Register with https:// should proceed to send the request."""
        from kernle.cli.commands.auth import cmd_auth

        mock_httpx = MagicMock()
        mock_response = MagicMock()
        mock_response.status_code = 201
        mock_response.json.return_value = {
            "user_id": "u123",
            "secret": "test-placeholder",
            "access_token": "tok",
            "expires_in": 604800,
        }
        mock_httpx.post.return_value = mock_response
        mock_httpx.ConnectError = Exception

        args = _make_register_args(backend_url="https://api.example.com")
        mock_k = MagicMock()
        mock_k.stack_id = "test-stack"

        with patch.dict("sys.modules", {"httpx": mock_httpx}):
            cmd_auth(args, k=mock_k)

        mock_httpx.post.assert_called_once()
