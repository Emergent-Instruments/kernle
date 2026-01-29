"""Shared helper functions for CLI commands."""

import json
import re
from typing import Any


def validate_input(value: str, field_name: str, max_length: int = 1000) -> str:
    """Validate and sanitize CLI inputs."""
    if not isinstance(value, str):
        raise ValueError(f"{field_name} must be a string")

    if len(value) > max_length:
        raise ValueError(f"{field_name} too long (max {max_length} characters)")

    # Remove null bytes and control characters except newlines
    sanitized = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", value)

    return sanitized


def print_json(data: Any) -> None:
    """Print data as formatted JSON."""
    print(json.dumps(data, indent=2, default=str))
