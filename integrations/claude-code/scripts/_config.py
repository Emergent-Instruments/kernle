"""Configuration and stack ID resolution for Claude Code hooks."""

import os
from pathlib import Path


def resolve_stack_id(cwd: str | None = None) -> str | None:
    """Resolve Kernle stack ID for Claude Code context.

    Resolution order:
    1. KERNLE_STACK_ID environment variable
    2. Project directory name from cwd
    3. None (let kernle CLI auto-resolve)
    """
    env_id = os.environ.get("KERNLE_STACK_ID")
    if env_id:
        return env_id

    if cwd:
        dir_name = Path(cwd).name
        if dir_name and dir_name not in ("workspace", "home", "/"):
            return dir_name

    return None


def get_config() -> dict:
    """Read plugin configuration from environment variables."""
    return {
        "kernle_bin": os.environ.get("KERNLE_BIN", "kernle"),
        "timeout": int(os.environ.get("KERNLE_TIMEOUT", "5")),
        "token_budget": int(os.environ.get("KERNLE_TOKEN_BUDGET", "8000")),
    }
