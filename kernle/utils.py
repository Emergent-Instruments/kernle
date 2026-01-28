"""
Kernle utilities - shared helper functions.
"""

import hashlib
import os
import platform
import subprocess
from typing import Optional


def _get_git_root() -> Optional[str]:
    """Get the root of the current git repository, if in one."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
        pass
    return None


def generate_default_agent_id() -> str:
    """Generate a default agent ID based on machine + project path.

    Combines:
    1. Machine identifier (hostname)
    2. Project path (git root or cwd)

    Returns a stable ID like 'auto-a1b2c3d4' that:
    - Same machine + same directory = same agent (consistent)
    - Different machine or path = different agent (isolated)

    The user can always override with explicit -a <name> or KERNLE_AGENT_ID env var.
    """
    # Get machine identifier
    machine = platform.node() or "unknown"

    # Get project path: prefer git root, fall back to cwd
    git_root = _get_git_root()
    project_path = git_root if git_root else os.getcwd()

    # Normalize the path for consistent hashing
    project_path = os.path.normpath(os.path.abspath(project_path))

    # Combine and hash
    identity_string = f"{machine}:{project_path}"
    hash_digest = hashlib.sha256(identity_string.encode()).hexdigest()

    return f"auto-{hash_digest[:8]}"


def resolve_agent_id(explicit_id: Optional[str] = None) -> str:
    """Resolve the agent ID with fallback chain.

    Resolution order:
    1. Explicit ID passed as argument (highest priority)
    2. KERNLE_AGENT_ID environment variable
    3. Auto-generated from machine + project path

    Args:
        explicit_id: Explicitly provided agent ID (e.g., from -a flag)

    Returns:
        The resolved agent ID
    """
    # 1. Explicit ID takes priority
    if explicit_id and explicit_id != "default":
        return explicit_id

    # 2. Check environment variable
    env_id = os.environ.get("KERNLE_AGENT_ID")
    if env_id:
        return env_id

    # 3. Generate from machine + project
    return generate_default_agent_id()
