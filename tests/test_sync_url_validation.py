"""Tests for _validate_backend_url in CLI sync module.

Verifies that the sync module's ``_validate_backend_url`` is wired to the
canonical ``validate_backend_url`` from ``kernle.core.validation`` and that
the import alias preserves the existing call-site contract.
"""

from kernle.cli.commands.sync import _validate_backend_url
from kernle.core.validation import validate_backend_url


class TestSyncValidateBackendUrlIsCanonical:
    """Verify sync's _validate_backend_url delegates to core.validation."""

    def test_sync_alias_is_canonical_function(self):
        """The sync module's _validate_backend_url must be the same object
        as kernle.core.validation.validate_backend_url."""
        assert _validate_backend_url is validate_backend_url


class TestValidateBackendUrl:
    """URL validation prevents bearer tokens leaking over plaintext HTTP."""

    def test_https_url_passes(self):
        assert _validate_backend_url("https://api.example.com") == "https://api.example.com"

    def test_http_localhost_passes(self):
        assert _validate_backend_url("http://localhost:8000") == "http://localhost:8000"

    def test_http_127_0_0_1_passes(self):
        assert _validate_backend_url("http://127.0.0.1:8000") == "http://127.0.0.1:8000"

    def test_http_localhost_no_port_passes(self):
        assert _validate_backend_url("http://localhost") == "http://localhost"

    def test_http_remote_rejected(self):
        assert _validate_backend_url("http://evil.example.com") is None

    def test_ftp_scheme_rejected(self):
        assert _validate_backend_url("ftp://example.com") is None

    def test_empty_string_returns_none(self):
        assert _validate_backend_url("") is None

    def test_none_returns_none(self):
        assert _validate_backend_url(None) is None

    def test_https_with_port_passes(self):
        assert _validate_backend_url("https://api.example.com:443") == "https://api.example.com:443"

    def test_http_remote_with_port_rejected(self):
        assert _validate_backend_url("http://evil.example.com:8080") is None
