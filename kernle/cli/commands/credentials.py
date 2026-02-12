"""Credential management helpers for Kernle CLI."""

import json
import sys
from pathlib import Path
from urllib.parse import urlparse

from kernle.utils import get_kernle_home


def get_credentials_path() -> Path:
    """Get the path to the credentials file."""
    return get_kernle_home() / "credentials.json"


def load_credentials():
    """Load credentials from ~/.kernle/credentials.json."""
    creds_path = get_credentials_path()
    if not creds_path.exists():
        return None
    try:
        with open(creds_path) as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError):
        return None


def save_credentials(credentials: dict):
    """Save credentials to ~/.kernle/credentials.json."""
    creds_path = get_credentials_path()
    creds_path.parent.mkdir(parents=True, exist_ok=True)
    with open(creds_path, "w") as f:
        json.dump(credentials, f, indent=2)
    # Set restrictive permissions (owner read/write only)
    creds_path.chmod(0o600)


def clear_credentials():
    """Remove the credentials file."""
    creds_path = get_credentials_path()
    if creds_path.exists():
        creds_path.unlink()
        return True
    return False


def _is_local_http(url: str) -> bool:
    """Check if a URL is a local development HTTP URL (localhost or 127.0.0.1).

    Uses proper URL parsing to prevent bypass via crafted hostnames like
    http://localhost.evil.com or http://localhost@evil.com.
    """
    try:
        parsed = urlparse(url)
        hostname = parsed.hostname  # Strips port, userinfo, etc.
        return hostname in ("localhost", "127.0.0.1")
    except Exception:
        return False


def prompt_backend_url(current_url: str = None) -> str:
    """Prompt user for backend URL."""
    default = current_url or "https://api.kernle.io"
    print(f"Backend URL [{default}]: ", end="", flush=True)
    try:
        url = input().strip()
        result = url if url else default
        # SECURITY: Warn if not using HTTPS (credentials would be sent in cleartext)
        if result and not result.startswith("https://"):
            if _is_local_http(result):
                pass  # Allow localhost for development
            else:
                print("⚠️  WARNING: Using non-HTTPS URL. Credentials will be sent in cleartext!")
                print("   This is insecure for production use. Press Ctrl+C to abort.")
        return result
    except (EOFError, KeyboardInterrupt):
        print("\nAborted.")
        sys.exit(1)


def warn_non_https_url(url: str, source: str = None) -> None:
    """Warn if using non-HTTPS URL (credentials would be sent in cleartext).

    Args:
        url: The backend URL to check
        source: Where the URL came from (e.g., "args", "env", "credentials") for context
    """
    if not url or url.startswith("https://"):
        return
    # Allow localhost for development (uses proper URL parsing)
    if _is_local_http(url):
        return
    source_msg = f" (from {source})" if source else ""
    print(f"⚠️  WARNING: Using non-HTTPS URL{source_msg}. Credentials will be sent in cleartext!")
    print("   This is insecure for production use.")


def require_https_url(url: str, source: str = None) -> None:
    """Block non-HTTPS, non-localhost URLs. Raises SystemExit.

    Args:
        url: The backend URL to check
        source: Where the URL came from (e.g., "args", "env", "credentials") for context
    """
    if not url or url.startswith("https://"):
        return
    # Allow localhost for development (uses proper URL parsing)
    if _is_local_http(url):
        return
    source_msg = f" (from {source})" if source else ""
    print(f"\n⚠  BLOCKED: Refusing to send credentials over plaintext HTTP{source_msg}")
    print(f"   URL: {url}")
    print("   Use https:// or http://localhost for development.")
    sys.exit(1)
